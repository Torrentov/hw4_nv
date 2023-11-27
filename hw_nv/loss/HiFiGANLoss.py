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

    def _discriminator_loss(self, disc_gen, disc_gt):
        loss = 0
        for i in range(len(disc_gen)):
            gt_loss = torch.mean(torch.square(disc_gt[i] - 1))
            gen_loss = torch.mean(torch.square(disc_gen[i]))
            loss += gt_loss + gen_loss
        return loss

    def _generator_loss(self, disc_gen):
        loss = 0
        for i in range(len(disc_gen)):
            loss += torch.mean(torch.square(disc_gen[i] - 1))
        return loss

    def _mel_loss(self, audio_gen, mel_gt):
        mel_gen = self.mel_spectrogram(audio_gen).squeeze(1)
        if mel_gen.size(2) > mel_gt.size(2):
            padding_size = mel_gen.size(2) - mel_gt.size(2)
            mel_gt = F.pad(mel_gt, (0, padding_size))
        return self.l1_loss(mel_gen, mel_gt)

    def _feature_matching_loss(self, features_gen, features_gt):
        loss = 0
        for i in range(len(features_gen)):
            for j in range(len(features_gen[i])):
                loss += self.l1_loss(features_gen[i][j], features_gt[i][j])
        return loss

    def discriminator_loss(self,
                           mpd_generated, mpd_ground_truth,
                           msd_generated, msd_ground_truth,
                           **batch):
        mpd_loss = self._discriminator_loss(mpd_generated,
                                            mpd_ground_truth)
        msd_loss = self._discriminator_loss(msd_generated,
                                            msd_ground_truth)
        return {
            "mpd_loss": mpd_loss,
            "msd_loss": msd_loss,
            "discriminator_loss": mpd_loss + msd_loss
        }

    def generator_loss(self,
                       mpd_generated, msd_generated,
                       audio_generated, mel_ground_truth,
                       mpd_features_generated, mpd_features_ground_truth,
                       msd_features_generated, msd_features_ground_truth,
                       **batch):
        mel_loss = self._mel_loss(audio_generated, mel_ground_truth)
        mel_loss = self.mel_loss_lambda * mel_loss

        generator_mpd_loss = self._generator_loss(mpd_generated)
        generator_msd_loss = self._generator_loss(msd_generated)
        generator_discriminator_loss = generator_msd_loss + generator_mpd_loss

        mpd_feature_matching_loss = self._feature_matching_loss(mpd_features_generated,
                                                                mpd_features_ground_truth)
        msd_feature_matching_loss = self._feature_matching_loss(msd_features_generated,
                                                                msd_features_ground_truth)
        feature_matching_loss = msd_feature_matching_loss + mpd_feature_matching_loss
        feature_matching_loss = self.fm_loss_lambda * feature_matching_loss

        total_generator_loss = generator_discriminator_loss + feature_matching_loss + mel_loss

        return {
            "generator_loss": total_generator_loss,
            "mel_loss": mel_loss,
            "generator_discriminator_loss": generator_discriminator_loss,
            "feature_matching_loss": feature_matching_loss,
            "generator_mpd_loss": generator_mpd_loss,
            "generator_msd_loss": generator_msd_loss,
            "mpd_feature_matching_loss": mpd_feature_matching_loss,
            "msd_feature_matching_loss": msd_feature_matching_loss
        }
