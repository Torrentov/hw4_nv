import unittest

import torch

from hw_nv.datasets import LJspeechDataset, CustomDirAudioDataset, CustomAudioDataset
from hw_nv.tests.utils import clear_log_folder_after_use
from hw_nv.utils import ROOT_PATH
from hw_nv.utils.parse_config import ConfigParser


class TestDataset(unittest.TestCase):
    def test_ljspeech(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            ds = LJspeechDataset(
                "train",
                config_parser=config_parser,
                max_audio_length=13,
                limit=10,
            )
            print(ds[0]['audio'].shape)
            print(ds[0]['spectrogram'].shape)
            self._assert_training_example_is_good(ds[0])

    def test_custom_dir_dataset(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            audio_dir = str(ROOT_PATH / "test_data" / "audio")
            transc_dir = str(ROOT_PATH / "test_data" / "transcriptions")

            ds = CustomDirAudioDataset(
                audio_dir,
                transc_dir,
                config_parser=config_parser,
                limit=10,
                max_audio_length=8,
            )
            self._assert_training_example_is_good(ds[0])

    def test_custom_dataset(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            audio_path = ROOT_PATH / "test_data" / "audio"
            transc_path = ROOT_PATH / "test_data" / "transcriptions"
            with (transc_path / "84-121550-0000.txt").open() as f:
                transcription = f.read().strip()
            data = [
                {
                    "path": str(audio_path / "84-121550-0001.flac"),
                },
                {
                    "path": str(audio_path / "84-121550-0000.flac"),
                    "text": transcription
                }
            ]

            ds = CustomAudioDataset(
                data=data,
                config_parser=config_parser,
            )
            self._assert_training_example_is_good(ds[0], contains_text=False)
            self._assert_training_example_is_good(ds[1])

    def _assert_training_example_is_good(self, training_example: dict, contains_text=True):

        for field, expected_type in [
            ("audio", torch.Tensor),
            ("spectrogram", torch.Tensor),
            ("duration", float),
            ("audio_path", str),
        ]:
            self.assertIn(field, training_example, f"Error during checking field {field}")
            self.assertIsInstance(training_example[field], expected_type,
                                  f"Error during checking field {field}")

        # check waveform dimensions
        batch_dim, audio_dim, = training_example["audio"].size()
        self.assertEqual(batch_dim, 1)
        self.assertGreater(audio_dim, 1)

        # check spectrogram dimensions
        batch_dim, freq_dim, time_dim = training_example["spectrogram"].size()
        self.assertEqual(batch_dim, 1)
        self.assertEqual(freq_dim, 80)
        self.assertGreater(time_dim, 1)