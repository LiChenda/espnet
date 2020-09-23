from functools import reduce
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.enh.nets.tf_mask_net_ctx import TFMaskingNetCTX
from espnet2.enh.nets.tf_mask_net_joint_ctx import TFMaskingNet_Joint_CTX
from espnet2.enh.nets.tasnet_ctx import TasNetCTX
from espnet2.enh.nets.ctx_predictor import CTXPredictor

import torch.nn.functional as F


class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
            self, enh_model: Optional[AbsEnhancement],
            ctx_predictor: Optional[CTXPredictor],
            use_pit: bool = True,
            ctx_mode: str = None,
            ctx_factor: float = 0.5,
    ):
        assert check_argument_types()

        super().__init__()
        self.ctx_mode = ctx_mode
        self.enh_model = enh_model
        self.ctx_factor = ctx_factor
        self.ctx_predictor = ctx_predictor
        self.num_spk = enh_model.num_spk
        self.num_noise_type = getattr(self.enh_model, "num_noise_type", 1)
        # get mask type for TF-domain models
        self.mask_type = getattr(self.enh_model, "mask_type", None)
        # get loss type for model training
        self.loss_type = getattr(self.enh_model, "loss_type", None)
        assert self.loss_type in (
            # mse_loss(predicted_mask, target_label)
            "mask_mse",
            # mse_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
            "magnitude",
            # l1_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
            "magnitude_l1",
            # mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
            "spectrum",
            "spectrum_l1",
            # si_snr(enhanced_waveform, target_waveform)
            "si_snr",
        ), self.loss_type
        # for multi-channel signal
        self.ref_channel = getattr(self.enh_model, "ref_channel", -1)
        self.use_pit = use_pit

    def _psm_theta(self, r, mix):
        eps = 1e-8
        phase_r = r / (abs(r) + eps)
        phase_mix = mix / (abs(mix) + eps)
        # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
        cos_theta = phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
        return cos_theta

    def _create_mask_label(self, mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

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
            "PSM^2",
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
                # TODO(Wangyou): need to fix this,
                #  as noise referecens are provided separately
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
            elif mask_type == "PSM^2":
                # This is for training beamforming masks
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                        phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r).pow(2) / (abs(mix_spec).pow(2) + eps)) * cos_theta
                mask = mask.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

    def forward(
            self,
            speech_mix: torch.Tensor,
            speech_mix_lengths: torch.Tensor = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or
            # (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        # dereverberated noisy signal
        # (optional, only used for frontend models with WPE)
        dereverb_speech_ref = kwargs.get("dereverb_ref", None)

        ctx_given = [
            kwargs.get("ctx_{}".format(spk + 1), None) for spk in range(self.num_spk)
        ]
        ctx_lengths = kwargs.get('ctx_1_lengths', None)
        if ctx_lengths is not None:
            ctx_given = [c[:, : ctx_lengths.max(), :] for c in ctx_given]

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # for data-parallel
        speech_ref = speech_ref[:, :, : speech_lengths.max()]
        speech_mix = speech_mix[:, : speech_lengths.max()]

        ctx_perm = None
        ctx_loss = torch.tensor(0)
        if self.ctx_mode in ["joint_train", "predict"]:
            ctx_pre, _ = self.ctx_predictor(speech_mix, speech_lengths)
            ctx_feed = ctx_pre
        else:
            ctx_feed = ctx_given
        if self.ctx_mode == "joint_train":
            ctx_loss, ctx_perm = self._permutation_loss(ctx_given, ctx_pre, self.ctx_loss)
            pass

        if self.loss_type != "si_snr":
            # prepare reference speech and reference spectrum
            speech_ref = torch.unbind(speech_ref, dim=1)
            spectrum_ref = [self.enh_model.stft(sr)[0] for sr in speech_ref]

            # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
            spectrum_ref = [
                ComplexTensor(sr[..., 0], sr[..., 1]) for sr in spectrum_ref
            ]
            spectrum_mix = self.enh_model.stft(speech_mix)[0]
            spectrum_mix = ComplexTensor(spectrum_mix[..., 0], spectrum_mix[..., 1])

            # predict separated speech and masks
            if isinstance(self.enh_model, TFMaskingNetCTX):
                spectrum_pre, tf_length, mask_pre = self.enh_model(
                    speech_mix, ctx_feed, speech_lengths
                )
            else:
                spectrum_pre, tf_length, mask_pre = self.enh_model(
                    speech_mix, speech_lengths
                )

            # compute TF masking loss
            if self.loss_type == "magnitude" or self.loss_type == 'magnitude_l1':
                # compute loss on magnitude spectrum
                loss_func = self.tf_l1_loss if ('_l1' in self.mask_type) else self.tf_mse_loss
                if "PSM" in self.mask_type:
                    thetas = [self._psm_theta(sr, spectrum_mix) for sr in spectrum_ref]
                else:
                    thetas = [1 for sr in spectrum_ref]
                magnitude_pre = [abs(ps + 1e-8) for ps in spectrum_pre]
                magnitude_ref = [abs(sr + 1e-8) * theta for sr, theta in zip(spectrum_ref, thetas)]
                tf_loss, perm = self._permutation_loss(
                    magnitude_ref, magnitude_pre, loss_func, use_pit=self.use_pit, perm=ctx_perm
                )
            elif self.loss_type == "spectrum" or self.loss_type == "spectrum_l1":
                # compute loss on complex spectrum
                loss_func = self.tf_l1_loss if ('_l1' in self.mask_type) else self.tf_mse_loss
                tf_loss, perm = self._permutation_loss(
                    spectrum_ref, spectrum_pre, loss_func, use_pit=self.use_pit, perm=ctx_perm
                )
            elif self.loss_type.startswith("mask"):
                if self.loss_type == "mask_mse":
                    loss_func = self.tf_mse_loss
                else:
                    raise ValueError("Unsupported loss type: %s" % self.loss_type)

                assert mask_pre is not None
                mask_pre_ = [
                    mask_pre["spk{}".format(spk + 1)] for spk in range(self.num_spk)
                ]

                # prepare ideal masks
                mask_ref = self._create_mask_label(
                    spectrum_mix, spectrum_ref, mask_type=self.mask_type
                )
                # compute TF masking loss
                tf_loss, perm = self._permutation_loss(mask_ref, mask_pre_, loss_func, use_pit=self.use_pit,
                                                       perm=ctx_perm)
            else:
                raise ValueError("Unsupported loss type: %s" % self.loss_type)

            if "noise1" in mask_pre:
                if noise_ref is None:
                    raise ValueError(
                        "No noise reference for training!\n"
                        'Please specify "--use_noise_ref true" in run.sh'
                    )

                noise_ref = torch.unbind(noise_ref, dim=1)
                noise_spectrum_ref = [
                    self.enh_model.stft(nr)[0] for nr in noise_ref
                ]
                noise_spectrum_ref = [
                    ComplexTensor(nr[..., 0], nr[..., 1])
                    for nr in noise_spectrum_ref
                ]
                noise_mask_ref = self._create_mask_label(
                    spectrum_mix, noise_spectrum_ref, mask_type=self.mask_type
                )

                mask_noise_pre = [
                    mask_pre["noise{}".format(n + 1)]
                    for n in range(self.num_noise_type)
                ]
                tf_noise_loss, perm_n = self._permutation_loss(
                    noise_mask_ref, mask_noise_pre, self.tf_mse_loss, use_pit=self.use_pit
                )
                tf_loss = tf_loss + tf_noise_loss
            if self.training:
                si_snr = None
            else:
                speech_pre = [
                    self.enh_model.stft.inverse(ps, speech_lengths)[0]
                    for ps in spectrum_pre
                ]
                if speech_ref[0].dim() == 3:
                    # For si_snr loss, only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.si_snr_loss, perm=perm, use_pit=self.use_pit
                )
                si_snr = -si_snr_loss.detach()

            loss = tf_loss
            stats = dict(si_snr=si_snr, loss=loss.detach(), tf_loss=tf_loss.detach())

        else:
            if speech_ref.dim() == 4:
                # For si_snr loss of multi-channel input,
                # only select one channel as the reference
                speech_ref = speech_ref[..., self.ref_channel]

            if isinstance(self.enh_model, TasNetCTX):
                speech_pre, speech_lengths, *__ = self.enh_model.forward_rawwav(
                    speech_mix, ctx_feed, ilens=speech_lengths
                )
            else:
                speech_pre, speech_lengths, *__ = self.enh_model.forward_rawwav(
                    speech_mix, speech_lengths
                )
            # speech_pre: list[(batch, sample)]
            assert speech_pre[0].dim() == 2, speech_pre[0].dim()
            speech_ref = torch.unbind(speech_ref, dim=1)

            if self.enh_model.predict_noise:
                # current only support single noise source
                assert len(speech_pre) - self.num_spk == 1
                predict_noise = [speech_pre[-1]]
                speech_pre = speech_pre[0:self.num_spk]
                noise_ref = torch.unbind(noise_ref, dim=1)
                noise_loss, _ = self._permutation_loss(noise_ref, predict_noise, self.si_snr_loss_zeromean)

            # compute si-snr loss
            si_snr_loss, perm = self._permutation_loss(
                speech_ref, speech_pre, self.si_snr_loss_zeromean, use_pit=self.use_pit, perm=ctx_perm
            )
            si_snr = -si_snr_loss
            loss = si_snr_loss
            stats = dict(si_snr=si_snr.detach())
            if self.enh_model.predict_noise:
                loss = loss + noise_loss
                stats['noise_snr'] = - noise_loss.detach()

        if self.ctx_mode == 'joint_train':
            ctx_loss = ctx_loss * 10
            loss = (1 - self.ctx_factor) * loss + self.ctx_factor * ctx_loss
            stats['ctx_loss'] = ctx_loss.detach()
        stats['loss'] = loss.detach()
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """

        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if isinstance(ref, ComplexTensor):
            eps = 1e-8
        else:
            eps = 0
        if ref.dim() == 3:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))

        return mseloss

    @staticmethod
    def ctx_loss(ref, inf):
        """time-frequency L1 loss.

        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        ref_len = ref.shape[1]
        inf_len = inf.shape[1]
        if ref_len > inf_len:
            assert ((ref_len - inf_len) / ref_len) < 0.2
            ref = ref[:, 0:inf_len, :]
        elif ref_len < inf_len:
            assert ((inf_len - ref_len) / inf_len) < 0.2
            inf = inf[:, 0:ref_len, :]

        l1loss = (abs(ref - inf) ** 2).mean(dim=[1, 2])

        return l1loss

    @staticmethod
    def tf_l1_loss(ref, inf):
        """time-frequency L1 loss.

        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if isinstance(ref, ComplexTensor):
            eps = 1e-8
        else:
            eps = 0
        if ref.dim() == 3:
            l1loss = abs(ref - inf + eps).mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = abs(ref - inf + eps).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))
        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """si-snr loss

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        eps = 1e-8
        ref = ref / (torch.norm(ref, p=2, dim=1, keepdim=True) + eps)
        inf = inf / (torch.norm(inf, p=2, dim=1, keepdim=True) + eps)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr

    @staticmethod
    def si_snr_loss_zeromean(ref, inf):
        """si_snr loss with zero-mean in pre-processing.

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        eps = 1e-8

        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + eps  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
                torch.sum(e_noise ** 2, dim=1) + eps
        )
        # print('pair_si_snr',pair_wise_si_snr[0,:])
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)  # [B]
        # print(pair_wise_si_snr)

        return -1 * pair_wise_si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None, use_pit=True):
        """The basic permutation loss function.

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

        if use_pit:
            losses = torch.stack(
                [pair_loss(p) for p in permutations(range(num_spk))], dim=1
            )
            if perm is None:
                loss, perm = torch.min(losses, dim=1)
            else:
                loss = losses[torch.arange(losses.shape[0]), perm]
        else:
            loss = pair_loss(list(range(num_spk)))
            perm = list(range(num_spk))

        return loss.mean(), perm

    def collect_feats(
            self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
