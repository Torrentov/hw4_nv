import torch.nn as nn
import torch
import torch.nn.functional as F

from hw_nv.preprocessing.mel_spectrogram import MelSpectrogram


class HiFiGANLoss(nn.Module):
    def __init__(self, fm_loss_lambda=2, mel_loss_lambda=45):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.mel_spectrogram = MelSpectrogram()
        self.fm_loss_lambda = fm_loss_lambda
        self.mel_loss_lambda = mel_loss_lambda

    def discriminator_loss(self, disc_gen, disc_gt):
        loss = 0
        for i in range(len(disc_gen)):
            gt_loss = torch.mean(torch.square(disc_gt[i] - 1))
            gen_loss = torch.mean(torch.square(disc_gen[i]))
            loss += gt_loss + gen_loss
        return loss

    def generator_loss(self, disc_gen):
        loss = 0
        for i in range(len(disc_gen)):
            loss += torch.mean(torch.square(disc_gen[i] - 1))
        return loss

    def mel_loss(self, audio_gen, mel_gt):
        mel_gen = self.mel_spectrogram(audio_gen).squeeze(1)
        if mel_gen.size(2) > mel_gt.size(2):
            padding_size = mel_gen.size(2) - mel_gt.size(2)
            mel_gt = F.pad(mel_gt, (0, padding_size))
        return self.l1_loss(mel_gen, mel_gt)

    def feature_matching_loss(self, features_gen, features_gt):
        loss = 0
        for i in range(len(features_gen)):
            for j in range(len(features_gen[i])):
                loss += self.l1_loss(features_gen[i][j], features_gt[i][j])
        return loss

    def forward(self,
                mpd_generated, mpd_ground_truth,
                msd_generated, msd_ground_truth,
                audio_generated, mel_ground_truth,
                mpd_features_generated, mpd_features_ground_truth,
                msd_features_generated, msd_features_ground_truth,
                **batch):
        mel_loss = self.mel_loss(audio_generated, mel_ground_truth)

        mpd_loss = self.discriminator_loss(mpd_generated,
                                           mpd_ground_truth)
        msd_loss = self.discriminator_loss(msd_generated,
                                           msd_ground_truth)
        discriminator_loss = mpd_loss + msd_loss

        generator_mpd_loss = self.generator_loss(mpd_generated)
        generator_msd_loss = self.generator_loss(msd_generated)
        generator_loss = generator_msd_loss + generator_mpd_loss

        mpd_feature_matching_loss = self.feature_matching_loss(mpd_features_generated,
                                                               mpd_features_ground_truth)
        msd_feature_matching_loss = self.feature_matching_loss(msd_features_generated,
                                                               msd_features_ground_truth)
        feature_matching_loss = msd_feature_matching_loss + mpd_feature_matching_loss

        total_generator_loss = generator_loss + \
                               self.fm_loss_lambda * feature_matching_loss + \
                               self.mel_loss_lambda * mel_loss
        total_discriminator_loss = discriminator_loss

        total_loss = total_discriminator_loss + total_generator_loss

        result = {
            "total_loss": total_loss,
            "total_generator_loss": total_generator_loss,
            "total_discriminator_loss": total_discriminator_loss,
            "mel_loss": mel_loss,
            "generator_loss": generator_loss,
            "feature_matching_loss": feature_matching_loss,
            "mpd_loss": mpd_loss,
            "msd_loss": msd_loss,
            "generator_mpd_loss": generator_mpd_loss,
            "generator_msd_loss": generator_msd_loss,
            "mpd_feature_matching_loss": mpd_feature_matching_loss,
            "msd_feature_matching_loss": msd_feature_matching_loss
        }
        return result

