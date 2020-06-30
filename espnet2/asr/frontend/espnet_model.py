from typing import Dict
from typing import Optional
from typing import Tuple
from itertools import permutations

import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from torch_complex.tensor import ComplexTensor
from functools import reduce


class ESPnetFrontendModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self, frontend: Optional[AbsFrontend],
    ):
        assert check_argument_types()

        super().__init__()

        self.frontend = frontend
        self.num_spk = frontend.num_spk
        self.num_noise_type = frontend.num_noise_type
        self.fs = frontend.fs
        self.tf_factor = frontend.tf_factor
        self.mask_type = frontend.mask_type
        # for multi-channel signal
        self.ref_channel = frontend.ref_channel

    def _create_mask_label(self, mix_spec, ref_spec, mask_type="IAM"):
        """
        :param mix_spec: ComplexTensor(B, T, F)
        :param ref_spec: [ComplexTensor(B, T, F), ...] or ComplexTensor(B, T, F)
        :param noise_spec: ComplexTensor(B, T, F)
        :return: [Tensor(B, T, F), ...] or [ComplexTensor(B, T, F), ...]
        """

        assert mask_type in [
            "IBM",
            "IRM",
            "IAM",
            "PSM",
            "NPSM",
            "ICM",
        ], f"mask type {mask_type} not supported"
        eps = 10e-8
        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [abs(r) >= abs(n) for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                # TODO (Wangyou): need to fix this, as noise referecens are provided separately
                mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + eps)
            elif mask_type == "IAM":
                mask = abs(r) / (abs(mix_spec) + eps)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "PSM" or mask_type == "NPSM":
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r) / (abs(mix_spec) + eps)) * cos_theta
                mask = (
                    mask.clamp(min=0, max=1)
                    if mask_label == "NPSM"
                    else mask.clamp(min=-1, max=1)
                )
            elif mask_type == "ICM":
                mask = r / (mix_spec + eps)
                mask.real = mask.real.clamp(min=-1, max=1)
                mask.imag = mask.imag.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

    def forward(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
            speech_lengths: (Batch,)
        """
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using frontend models with beamformering)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        # dereverberated noisy signal (optional, only used for frontend models with WPE)
        dereverb_speech_ref = kwargs.get("dereverb_ref", None)

        speech_lengths = speech_mix_lengths
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # for data-parallel
        if speech_ref.dim() == 3:  # single-channel
            speech_ref = speech_ref[:, :, : speech_lengths.max()]
        else:  # multi-channel
            speech_ref = speech_ref[:, :, : speech_lengths.max(), :]
        if speech_mix.dim() == 3:  # single-channel
            speech_mix = speech_mix[:, : speech_lengths.max()]
        else:  # multi-channel
            speech_mix = speech_mix[:, : speech_lengths.max(), :]

        if self.tf_factor > 0:
            # prepare reference speech and reference spectrum
            speech_ref = torch.unbind(speech_ref, dim=1)
            spectrum_ref = [self.frontend.stft(sr)[0] for sr in speech_ref]

            # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
            spectrum_ref = [
                ComplexTensor(sr[..., 0], sr[..., 1]) for sr in spectrum_ref
            ]
            spectrum_mix = self.frontend.stft(speech_mix)[0]
            spectrum_mix = ComplexTensor(spectrum_mix[..., 0], spectrum_mix[..., 1])

            # prepare ideal masks
            mask_ref = self._create_mask_label(
                spectrum_mix, spectrum_ref, mask_type=self.mask_type
            )

            if dereverb_speech_ref is not None:
                dereverb_spectrum_ref = self.frontend.stft(dereverb_speech_ref)[0]
                dereverb_spectrum_ref = ComplexTensor(
                    dereverb_spectrum_ref[..., 0], dereverb_spectrum_ref[..., 1]
                )
                # ComplexTensor(B, T, F) or ComplexTensor(B, T, C, F)
                dereverb_mask_ref = self._create_mask_label(
                    spectrum_mix, [dereverb_spectrum_ref], mask_type=self.mask_type
                )[0]

            if noise_ref is not None:
                noise_ref = torch.unbind(noise_ref, dim=1)
                noise_spectrum_ref = [self.frontend.stft(nr)[0] for nr in noise_ref]
                noise_spectrum_ref = [
                    ComplexTensor(nr[..., 0], nr[..., 1]) for nr in noise_spectrum_ref
                ]
                noise_mask_ref = self._create_mask_label(
                    spectrum_mix, noise_spectrum_ref, mask_type=self.mask_type
                )

            # predict separated speech and masks
            spectrum_pre, tf_length, mask_pre = self.frontend(
                speech_mix, speech_lengths
            )

            # TODO:Chenda, Shall we add options for computing loss on
            #  the masked spectrum?
            # compute TF masking loss
            if mask_pre is None:
                # compute loss on magnitude spectrum instead
                magnitude_pre = [abs(ps) for ps in spectrum_pre]
                magnitude_ref = [abs(sr) for sr in spectrum_ref]
                tf_loss, perm = self._permutation_loss(
                    magnitude_ref, magnitude_pre, self.tf_mse_loss
                )
            else:
                mask_pre_ = [
                    mask_pre["spk{}".format(spk + 1)] for spk in range(self.num_spk)
                ]

                # compute TF masking loss
                # TODO:Chenda, Shall we add options for computing loss on the masked spectrum?
                tf_loss, perm = self._permutation_loss(
                    mask_ref, mask_pre_, self.tf_mse_loss
                )

                if "dereverb" in mask_pre:
                    if dereverb_speech_ref is None:
                        raise ValueError(
                            "No dereverberated reference for training!\n"
                            'Please specify "--use_dereverb_ref true" in run.sh'
                        )
                    tf_loss = (
                        tf_loss
                        + self.tf_l1_loss(
                            dereverb_mask_ref, mask_pre["dereverb"]
                        ).mean()
                    )

                if "noise1" in mask_pre:
                    if noise_ref is None:
                        raise ValueError(
                            "No noise reference for training!\n"
                            'Please specify "--use_noise_ref true" in run.sh'
                        )
                    mask_noise_pre = [
                        mask_pre["noise{}".format(n + 1)]
                        for n in range(self.num_noise_type)
                    ]
                    tf_noise_loss, perm_n = self._permutation_loss(
                        noise_mask_ref, mask_noise_pre, self.tf_mse_loss
                    )
                    tf_loss = tf_loss + tf_noise_loss

            if self.tf_factor == 1.0:
                si_snr_loss = None
                si_snr = None
                loss = tf_loss
            else:
                speech_pre = [
                    self.frontend.stft.inverse(ps, speech_lengths)[0]
                    for ps in spectrum_pre
                ]
                if speech_ref.dim() == 4:
                    # For si_snr loss, only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.si_snr_loss, perm=perm
                )
                si_snr = -si_snr_loss

                loss = (1 - self.tf_factor) * si_snr_loss + self.tf_factor * tf_loss

            stats = dict(
                si_snr=si_snr.detach() if si_snr is not None else None,
                tf_loss=tf_loss.detach(),
                loss=loss.detach(),
            )
        else:
            # TODO:Jing, should find better way to configure for the choice of tf loss and time-only loss.
            if speech_ref.dim() == 4:
                # For si_snr loss of multi-channel input, only select one channel as the reference
                speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]

            speech_pre, speech_lengths, *__ = self.frontend.forward_rawwav(
                speech_mix, speech_lengths
            )
            speech_pre = torch.unbind(speech_pre, dim=1)

            # compute si-snr loss
            si_snr_loss, perm = self._permutation_loss(
                speech_ref, speech_pre, self.si_snr_loss
            )
            si_snr = -si_snr_loss
            loss = si_snr_loss
            stats = dict(si_snr=si_snr.detach(), loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @staticmethod
    def tf_mse_loss(ref, inf):
        """
        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            mseloss = ((ref - inf) ** 2).mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = ((ref - inf) ** 2).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))

        return mseloss

    @staticmethod
    def tf_l1_loss(ref, inf):
        """
        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            l1loss = abs(ref - inf).mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = abs(ref - inf).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))
        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """
        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """
        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            torch.Tensor: (batch)
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack(
            [pair_loss(p) for p in permutations(range(num_spk))], dim=1
        )
        if perm is None:
            loss, perm = torch.min(losses, dim=1)
        else:
            loss = losses[torch.arange(losses.shape[0]), perm]

        return loss.mean(), perm

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}