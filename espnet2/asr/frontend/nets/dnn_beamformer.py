from distutils.version import LooseVersion
from typing import Tuple

import torch
from torch.nn import functional as F
from torch_complex import functional as FC

from espnet.nets.pytorch_backend.frontends.beamformer import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer import (
    get_power_spectral_density_matrix,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from espnet2.asr.frontend.nets.conv_beamformer import get_covariances
from espnet2.asr.frontend.nets.conv_beamformer import get_WPD_filter_v2
from espnet2.asr.frontend.nets.conv_beamformer import perform_WPD_filtering
from torch_complex.tensor import ComplexTensor

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")
is_torch_1_3_plus = LooseVersion(torch.__version__) >= LooseVersion("1.3.0")


class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783

    """

    def __init__(
        self,
        bidim,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        bnmask: int = 2,
        dropout_rate: float = 0.0,
        badim: int = 320,
        ref_channel: int = -1,
        beamformer_type: str = "mvdr",
        eps: float = 1e-7,
        # only for WPD beamformer
        btaps: int = 5,
        bdelay: int = 3,
    ):
        super().__init__()
        self.mask = MaskEstimator(
            btype, bidim, blayers, bunits, bprojs, dropout_rate, nmask=bnmask
        )
        self.ref = AttentionReference(bidim, badim)
        self.ref_channel = ref_channel

        self.nmask = bnmask

        if beamformer_type not in ('mvdr', 'mpdr', 'wpd'):
            raise ValueError(
                "Not supporting beamformer_type={}".format(beamformer_type)
            )
        self.beamformer_type = beamformer_type
        self.eps = eps
        self.btaps = btaps
        self.bdelay = bdelay

    def forward(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)
            masks (torch.Tensor): (B, T, C, F)
        """

        def apply_beamforming(data, ilens, psd_speech, psd_n, beamformer_type):
            # u: (B, C)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_speech, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(
                    *(data.size()[:-3] + (data.size(-2),)), device=data.device
                )
                u[..., self.ref_channel].fill_(1)

            if beamformer_type in ('mpdr', 'mvdr'):
                ws = get_mvdr_vector(psd_speech, psd_n, u)
                enhanced = apply_beamforming_vector(ws, data)
            elif beamformer_type == 'wpd':
                ws = get_WPD_filter_v2(psd_speech, psd_n, u)
                enhanced = perform_WPD_filtering(ws, data, self.bdelay, self.btaps)
            else:
                raise ValueError('Not supporting beamformer_type={}'.format(beamformer_type))

            return enhanced, ws

        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)

        # mask: (B, F, C, T)
        masks, _ = self.mask(data, ilens)
        assert self.nmask == len(masks)

        if self.nmask == 2:  # (mask_speech, mask_noise)
            mask_speech, mask_noise = masks

            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            if self.beamformer_type == 'mvdr':
                # psd of noise
                psd_n = get_power_spectral_density_matrix(data, mask_noise)
            elif self.beamformer_type == 'mpdr':
                # psd of observed signal
                psd_n = FC.einsum('...ct,...et->...ce', [data, data.conj()])
            elif self.beamformer_type == 'wpd':
                # Calculate power: (..., C, T)
                power_speech = (data.real ** 2 + data.imag ** 2) * mask_speech
                # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
                power_speech = power_speech.mean(dim=-2)
                inverse_power = 1 / torch.clamp(power_speech, min=self.eps)
                # covariance of expanded observed speech
                psd_n = get_covariances(data, inverse_power, self.bdelay, self.btaps, get_vector=False)
            else:
                raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))

            enhanced, ws = apply_beamforming(data, ilens, psd_speech, psd_n, self.beamformer_type)

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(-1, -2)
            mask_speech = mask_speech.transpose(-1, -3)
        else:  # multi-speaker case: (mask_speech1, ..., mask_noise)
            mask_speech = list(masks[:-1])
            mask_noise = masks[-1]

            psd_speeches = [
                get_power_spectral_density_matrix(data, mask) for mask in mask_speech
            ]
            if self.beamformer_type == 'mvdr':
                # psd of noise
                psd_n = get_power_spectral_density_matrix(data, mask_noise)
            elif self.beamformer_type == 'mpdr':
                # psd of observed speech
                psd_n = FC.einsum('...ct,...et->...ce', [data, data.conj()])
            elif self.beamformer_type == 'wpd':
                # Calculate power: (..., C, T)
                power = data.real ** 2 + data.imag ** 2
                power_speeches = [power * mask for mask in mask_speech]
                # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
                power_speeches = [ps.mean(dim=-2) for ps in power_speeches]
                inverse_poweres = [1 / torch.clamp(ps, min=self.eps) for ps in power_speeches]
                # covariance of expanded observed speech
                psd_n = [
                    get_covariances(data, inv_ps, self.bdelay, self.btaps, get_vector=False)
                    for inv_ps in inverse_poweres
                ]
            else:
                raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))

            enhanced = []
            ws = []
            for i in range(self.nmask - 1):
                psd_speech = psd_speeches.pop(i)
                # treat all other speakers' psd_speech as noises
                if self.beamformer_type == 'mvdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_speech, sum(psd_speeches) + psd_n, self.beamformer_type
                    )
                elif self.beamformer_type == 'mpdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_speech, psd_n, self.beamformer_type
                    )
                elif self.beamformer_type == 'wpd':
                    enh, w = apply_beamforming(
                        data, ilens, psd_speech, psd_n[i], self.beamformer_type
                    )
                else:
                    raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))
                psd_speeches.insert(i, psd_speech)

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(-1, -2)
                mask_speech[i] = mask_speech[i].transpose(-1, -3)

                enhanced.append(enh)
                ws.append(w)

        return enhanced, ilens, masks


class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(
        self, psd_in: ComplexTensor, ilens: torch.LongTensor, scaling: float = 2.0
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        datatype = torch.bool if is_torch_1_3_plus else torch.uint8
        datatype2 = torch.bool if is_torch_1_2_plus else torch.uint8
        psd = psd_in.masked_fill(
            torch.eye(C, dtype=datatype, device=psd_in.device).type(datatype2), 0
        )
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real ** 2 + psd.imag ** 2) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens