# modules

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_complex.tensor import ComplexTensor

from espnet2.enh.long_seq_nets.rnns import *
from espnet2.enh.long_seq_nets.transformers import *
from espnet2.layers.stft import Stft
from espnet2.enh.abs_enh import AbsEnhancement
from collections import OrderedDict


# base module for DPRNN-related modules
class RNN_base(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer=4, bidirectional=True, model='LocalRNN', embedding='rnn',
                 attention_dim=256, num_head=4, att_ff=1024, layer_per_block=1, layer_type="transformer",
                 downsample_local=None, block_size=100,
                 hpooling=True, dropout=0):
        super(RNN_base, self).__init__()

        assert model in ['GlobalRNN', 'GlobalATT', 'LocalRNN', 'HRNN', 'LocalTransformer', 'GlobalTransformer',
                         'GlobalTransformerV2',
                         'SPKRNN', 'SPKRNN_O'], "model can only be 'GlobalRNN', 'LocalRNN' or 'HRNN' "
        self.model = model

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer

        # layers
        self.layers = nn.ModuleList([])
        for i in range(num_layer):
            if model == "GlobalATT":
                self.layers.append(
                    GlobalATT(self.input_dim, self.hidden_dim, bidirectional=bidirectional,
                              attention_dim=attention_dim, num_head=num_head, att_ff=att_ff,
                              dropout=dropout))
            elif model in ['LocalTransformer', 'GlobalTransformer', 'GlobalTransformerV2']:
                self.layers.append(
                    getattr(sys.modules[__name__], model)(self.input_dim, dropout=dropout, num_blocks=layer_per_block,
                                                          num_head=num_head, att_ff=att_ff,
                                                          downsample_local=downsample_local,
                                                          idx=i, layer_type=layer_type, block_size=block_size)
                )
            else:
                self.layers.append(
                    getattr(sys.modules[__name__], model)(self.input_dim, self.hidden_dim, bidirectional=bidirectional,
                                                          dropout=dropout))

    def pad_segment(self, input, segment_size):
        # input is the 2-D features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        if seq_len % segment_size != 0:
            rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
            if rest > 0:
                pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type()).to(input.device)
                input = torch.cat([input, pad], 2)
        else:
            rest = 0

        # extra padding at first and last segments
        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type()).to(input.device)
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_segment(self, input, segment_size):
        # split the sequence into segments
        # input is the 2-D sequence: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        num_segment = (seq_len - segment_size) // segment_stride + 1

        segments = [input[:, :, i * segment_stride:i * segment_stride + segment_size].unsqueeze(3) for i in
                    range(num_segment)]
        segments = torch.cat(segments, 3)  # B, N, segment, num_segment

        return segments.contiguous(), rest

    def merge_segment(self, input, rest):
        # merge the segments into sequence
        # input is the 3-D segments: (B, N, segment, num_segment)

        batch_size, dim, segment_size, num_segment = input.shape
        segment_stride = segment_size // 2

        output = torch.zeros(batch_size, dim, (num_segment - 1) * segment_stride + segment_size).type(input.type()).to(
            input.device)  # B, N, T

        for i in range(num_segment):
            output[:, :, i * segment_stride:i * segment_stride + segment_size] = \
                output[:, :, i * segment_stride:i * segment_stride + segment_size] + input[:, :, :, i]

        output = output[:, :, segment_stride:-rest - segment_stride]

        return output.contiguous()  # B, N, T

    def forward(self, input, spk_id=None):
        # assume that the input is already properly transformed (splitted, permuted, etc.)

        output = input
        for i in range(self.num_layer):
            if 'SPKRNN' in self.model and i == 1:
                output = self.layers[i](output, spk_id)
            else:
                output = self.layers[i](output)

        if isinstance(output, tuple):
            output = output[0]

        return output


class LongSeqMasking(AbsEnhancement):
    def __init__(self, n_fft=512, hop_length=160, window_size=400, feature_dim=256, hidden_dim=128, layer=4,
                 num_spk=2, block_size=200, sr=16000, bidirectional=True, layer_per_block=1, model='LocalRNN',
                 embedding='rnn',
                 attention_dim=256, num_head=4, att_ff=1024, downsample_local=None,
                 layer_type="transformer",
                 hpooling=True, dropout=0.0,
                 loss_type='magnitude', mask='relu', stitching_loss=False):
        """
        DPRNN-based T-F masking model for single-channel separation.
        args:
            n_fft: int, size of Fourier transform
            hop_length: int, the distance between neighboring sliding window
            frames.
            feature_dim: int, feature dimension for the input to the separator.
            hidden_dim: int, number of hidden units in each RNN in DPRNN blocks.
            layer: int, number of DPRNN layers.
            num_spk: int, number of speakers to separate.
            block_size: int, number of frames in each block. Each waveform is splitted into smaller blocks.
                    If block_size <= 0, then use the entire utterance as one block (no segmentation).
            sr: int, waveform sample rate.
            bidirectional: bool, causal or noncausal configuration for DPRNN.
            model: string, determine which RNN variant to use for the separator.
            hpooling: string, determine the pooling method for HRNN. Only valid for HRNN models.
            embedding: string, determine the way embeddings are processed in HRNN. Only valid for HRNN models.
        input:
            input: a batch of waveforms with shape (B, T).
        output:
            output: a batch of separated outputs with shape (B, num_block, C, T), where C is num_spk.
        """
        super(LongSeqMasking, self).__init__()

        assert mask in ['relu', 'sigmoid']

        self.enc_dim = n_fft // 2 + 1
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.stride_size = hop_length

        self.num_layer = layer
        self.num_spk = num_spk

        self.block_size = block_size
        self.loss_type = loss_type

        self.model = model

        self.eps = torch.finfo(torch.float32).eps

        # stft Encoder
        self.stft = Stft(n_fft=n_fft, hop_length=hop_length, win_length=window_size, center=False)

        self.enc_LN = nn.LayerNorm([self.enc_dim, self.block_size])
        # self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=self.eps)
        # bottleneck layer
        self.enc_BN = nn.Conv1d(self.enc_dim, self.feature_dim, 1)
        # separator
        self.separator = RNN_base(self.feature_dim, self.hidden_dim, num_layer=self.num_layer,
                                  bidirectional=bidirectional, model=self.model, embedding=embedding,
                                  attention_dim=attention_dim, num_head=num_head, att_ff=att_ff, layer_type=layer_type,
                                  downsample_local=downsample_local,
                                  hpooling=hpooling, dropout=dropout, layer_per_block=layer_per_block, block_size=block_size)

        # mask estimation layer
        self.mask = nn.Sequential(nn.Conv2d(self.feature_dim, self.enc_dim * self.num_spk, 1),
                                  nn.ReLU() if mask == 'relu' else nn.Sigmoid()
                                  )
        self.stitching_loss = stitching_loss

    def pad_waveform(self, input, window, stride):
        # zero-padding waveform according to window/stride size.
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type()).to(input.device)
            input = torch.cat([input, pad], 1)

        # extra padding at first and last segments
        pad_aux = Variable(torch.zeros(batch_size, stride)).type(input.type()).to(input.device)
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def segmentation(self, input):
        # only apply segmentation to waveform without separation
        # this is used to create matched block-level training targets
        # input shape: (B, T)

        batch_size = input.shape[0]
        # print(input.shape)
        # waveform padding
        output, pad_rest = self.pad_waveform(input, self.window_size, self.stride_size)  # B, T

        # use identity matrix for waveform encoder
        encoder_weight = torch.eye(self.window_size).type(input.type()).unsqueeze(1).to(input.device)
        enc_output = F.conv1d(output.unsqueeze(1), encoder_weight, stride=self.stride_size)  # B, N, L

        if self.block_size > 0:
            # split the encoder output into smaller blocks
            enc_blocks, _ = self.separator.split_segment(enc_output, self.block_size)  # B, N, block, num_block
            num_block = enc_blocks.shape[-1]
        else:
            # one block for entire utterance
            enc_blocks = enc_output.unsqueeze(3)
            num_block = 1

        enc_blocks = enc_blocks.permute(0, 3, 1, 2).contiguous().view(batch_size * num_block, self.window_size,
                                                                      self.block_size)  # B*num_block, N, block

        # decode back to waveforms
        output = F.conv_transpose1d(enc_blocks, encoder_weight, stride=self.stride_size)  # B*num_block, 1, L
        output = output.view(batch_size, num_block, -1)  # B, num_block, L

        return output

    def forward(self, input, ilens):
        # input shape: (B, block_num, T)

        batch_size = input.shape[0]

        blocked_wav = self.segmentation(input)  # B, block_num, L
        num_block, i_samples = blocked_wav.shape[1], blocked_wav.shape[2]
        blocked_wav = blocked_wav.view(-1, i_samples)

        # waveform encoder
        stft = self.stft(blocked_wav)[0]
        stft = ComplexTensor(stft[..., 0], stft[..., 1])  # ComplexTensor (batch*num_block, block, F)
        seq_len = stft.shape[-2]

        enc_blocks = abs(stft).permute(0, 2, 1)  # (B * num_block, F, block)

        # normalize encoder output and pass to bottleneck layer
        enc_blocks_feature = self.enc_BN(self.enc_LN(enc_blocks))  # B * num_blocks, H, block
        enc_blocks_feature = enc_blocks_feature.view(batch_size, num_block,
                                                     self.feature_dim, seq_len).permute(0, 2, 3,
                                                                                        1)  # B, N, block, num_block

        separate_blocks = self.separator(enc_blocks_feature)  # B, N, block, num_block

        # there's no need to overlap-and-add the blocks

        masks = self.mask(separate_blocks).view(batch_size, self.num_spk, self.enc_dim, seq_len,
                                                num_block)  # B, C, F, block, num_block
        masks = masks.permute(0, 4, 1, 3, 2).contiguous()  # [B, num_block, C, block, F]
        masks = masks.view(batch_size * num_block, self.num_spk, seq_len, self.enc_dim).unbind(dim=1)
        stft = stft.view(batch_size * num_block, seq_len, self.enc_dim)  # B, num_block, block, F
        masked_output = [stft * m for m in masks]  # B, num_block, block, F

        masks = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        return masked_output, ilens, masks
        # masked_output ComplexTensor (B * num_block , num_spk, block, F)

    def forward_rawwav(
            self, input: torch.Tensor, ilens: torch.Tensor
    ):
        predicted_spectrums, flens, masks = self.forward(input, ilens)
        with torch.no_grad():
            b, num_seg, L = self.segmentation(input).shape
            ilens = torch.tensor([L] * (b * num_seg))
        if predicted_spectrums is None:
            predicted_wavs = None
        elif isinstance(predicted_spectrums, list):
            # multi-speaker input
            predicted_wavs = [
                self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums
            ]
        else:
            # single-speaker input
            predicted_wavs = self.stft.inverse(predicted_spectrums, ilens)[0]

        return predicted_wavs, ilens, masks


if __name__ == '__main__':
    # from torchviz import make_dot

    input = torch.rand((1, 16000 * 30)).cuda()
    net = LongSeqMasking(block_size=150, model='GlobalTransformer', n_fft=512,
                         hop_length=256, window_size=512, layer=8, feature_dim=256, downsample_local=True,
                         att_ff=2048,
                         num_head=4,
                         layer_per_block=1).cuda()
    print(net)
    output, _, _ = net.forward(input, None)

    # dot = make_dot(output[0].abs(), params=dict(net.named_parameters()))
    # print(dot)
    # dot.render("/mnt/lustre/sjtu/home/cdl54/dot.png")
