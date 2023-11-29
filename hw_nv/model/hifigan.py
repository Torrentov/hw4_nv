import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from hw_nv.base import BaseModel


class ResBlockV1(nn.Module):
    def __init__(self, channels, kernel_size, dilation=((1, 1), (3, 1), (5, 1))):
        super().__init__()
        self.first_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation[i][0],
                      stride=1, padding=((kernel_size - 1) * dilation[i][0]) // 2)
            for i in range(len(dilation))
        ])

        self.second_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation[i][1],
                      stride=1, padding=((kernel_size - 1) * dilation[i][1]) // 2)
            for i in range(len(dilation))
        ])

        self.num_blocks = len(dilation)

    def forward(self, x):
        for i in range(self.num_blocks):
            output = F.leaky_relu(x)
            output = self.first_convs[i](output)
            output = F.leaky_relu(output)
            output = self.second_convs[i](output)
            x = x + output
        return x


class MultiRecepticeFieldFusion(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlockV1(channels, kernel_sizes[i], dilations[i])
            for i in range(len(kernel_sizes))
        ])

    def forward(self, x):
        results = self.resblocks[0](x)
        for i in range(1, len(self.resblocks)):
            results = results + self.resblocks[i](x)
        return results


class Generator(nn.Module):
    def __init__(self, channels, upsample_kernel_sizes, mrf_kernel_sizes, mrf_dilations):
        super().__init__()
        self.first_conv = nn.Conv1d(80, channels, 7, 1, padding=3)
        main_block = []
        current_channels = channels * 2
        next_channels = channels
        for i in range(len(upsample_kernel_sizes)):
            current_channels //= 2
            next_channels //= 2
            ct_stride = upsample_kernel_sizes[i] // 2
            ct_padding = (upsample_kernel_sizes[i] - ct_stride) // 2
            main_block.append(nn.LeakyReLU())
            main_block.append(nn.ConvTranspose1d(current_channels, next_channels,
                                                 upsample_kernel_sizes[i], stride=ct_stride, padding=ct_padding))
            main_block.append(MultiRecepticeFieldFusion(next_channels, mrf_kernel_sizes, mrf_dilations))
        self.main_block = nn.ModuleList(main_block)
        self.finisher = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(next_channels, 1, 7, 1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.first_conv(x)
        for module in self.main_block:
            x = module(x)
        x = self.finisher(x)
        return x


class SubMPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.main_convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 64, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(64, 128, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 256, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(256, 512, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), stride=1, padding=(2, 0)))
        ])
        self.finisher = weight_norm(nn.Conv2d(1024, 1, (3, 1), stride=1, padding=(2, 0)))

    def forward(self, x):
        features = []
        needed_padding = self.period - x.size(2) % self.period
        x = F.pad(x, (0, needed_padding), "reflect")
        x = x.reshape(x.size(0), 1, x.size(2) // self.period, self.period)
        for module in self.main_convs:
            x = module(x)
            x = F.leaky_relu(x)
            features.append(x)
        x = self.finisher(x)
        features.append(x)

        return x, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([
            SubMPD(2),
            SubMPD(3),
            SubMPD(5),
            SubMPD(7),
            SubMPD(11)
        ])

    def forward(self, audio_gen, audio_gt):
        output_gen = []
        features_gen = []
        output_gt = []
        features_gt = []
        for module in self.sub_discriminators:
            out_gen, feat_gen = module(audio_gen)
            out_gt, feat_gt = module(audio_gt)
            output_gen.append(out_gen)
            features_gen.append(feat_gen)
            output_gt.append(out_gt)
            features_gt.append(feat_gt)
        return output_gen, features_gen, output_gt, features_gt


class SubMSD(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        if use_spectral_norm:
            norm = spectral_norm
        else:
            norm = weight_norm
        self.main_convs = nn.ModuleList([
            norm(nn.Conv1d(1, 16, 15, stride=1, padding=7)),
            norm(nn.Conv1d(16, 64, 41, stride=4, groups=4, padding=20)),
            norm(nn.Conv1d(64, 256, 41, stride=4, groups=16, padding=20)),
            norm(nn.Conv1d(256, 1024, 41, stride=4, groups=64, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, stride=4, groups=256, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, stride=1, padding=20))
        ])
        self.finisher = norm(nn.Conv1d(1024, 1, 3, stride=1, padding=1))

    def forward(self, x):
        features = []
        for module in self.main_convs:
            x = module(x)
            x = F.leaky_relu(x)
            features.append(x)
        x = self.finisher(x)
        features.append(x)

        return x, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([
            SubMSD(use_spectral_norm=True),
            SubMSD(),
            SubMSD()
        ])

        self.data_preprocessing = nn.ModuleList([
            nn.AvgPool1d(4, stride=2, padding=2),
            nn.AvgPool1d(4, stride=2, padding=2)
        ])

    def forward(self, audio_gen, audio_gt):
        output_gen = []
        features_gen = []
        output_gt = []
        features_gt = []
        for i in range(len(self.sub_discriminators)):
            if i > 0:
                audio_gen = self.data_preprocessing[i - 1](audio_gen)
                audio_gt = self.data_preprocessing[i - 1](audio_gt)
            out_gen, feat_gen = self.sub_discriminators[i](audio_gen)
            out_gt, feat_gt = self.sub_discriminators[i](audio_gt)
            output_gen.append(out_gen)
            features_gen.append(feat_gen)
            output_gt.append(out_gt)
            features_gt.append(feat_gt)
        return output_gen, features_gen, output_gt, features_gt


class HiFiGAN(BaseModel):
    def __init__(self,
                 generator_channels, upsample_kernel_sizes, mrf_kernel_sizes, mrf_dilations,
                 **batch):
        super().__init__(**batch)
        self.generator = Generator(generator_channels, upsample_kernel_sizes, mrf_kernel_sizes, mrf_dilations)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward_generator(self, mel, **batch):
        audio_gen = self.generator(mel)
        return {"audio_generated": audio_gen,
                "mel_ground_truth": mel}

    def forward_discriminator(self, audio_generated, audio_gt, save_data=False, **batch):
        if audio_generated.size(2) > audio_gt.size(2):
            padding_size = audio_generated.size(2) - audio_gt.size(2)
            audio_gt = F.pad(audio_gt, (0, padding_size))
        if audio_gt.size(2) > audio_generated.size(2):
            padding_size = audio_gt.size(2) - audio_generated.size(2)
            audio_generated = F.pad(audio_generated, (0, padding_size))
        if save_data:
            mpd_out_gen, mpd_feat_gen, mpd_out_gt, mpd_feat_gt = self.mpd(audio_generated.detach(), audio_gt)
            msd_out_gen, msd_feat_gen, msd_out_gt, msd_feat_gt = self.msd(audio_generated.detach(), audio_gt)
        else:
            mpd_out_gen, mpd_feat_gen, mpd_out_gt, mpd_feat_gt = self.mpd(audio_generated, audio_gt)
            msd_out_gen, msd_feat_gen, msd_out_gt, msd_feat_gt = self.msd(audio_generated, audio_gt)
        result = {
            "audio_generated": audio_generated,
            "mpd_generated": mpd_out_gen,
            "mpd_features_generated": mpd_feat_gen,
            "mpd_ground_truth": mpd_out_gt,
            "mpd_features_ground_truth": mpd_feat_gt,
            "msd_generated": msd_out_gen,
            "msd_features_generated": msd_feat_gen,
            "msd_ground_truth": msd_out_gt,
            "msd_features_ground_truth": msd_feat_gt
        }
        return result

    def forward(self, mel, audio_gt=None, **batch):
        gen = self.forward_generator(mel)
        if audio_gt is not None:
            return self.forward_discriminator(**gen, audio_gt=audio_gt, **batch)
        return gen

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
