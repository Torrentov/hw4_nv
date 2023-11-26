import unittest

from tqdm import tqdm

from hw_nv.collate_fn.collate import collate_fn
from hw_nv.datasets import LJspeechDataset
from hw_nv.tests.utils import clear_log_folder_after_use
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser


class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            ds = LJspeechDataset(
                "train",
                config_parser=config_parser
            )

            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            self.assertIn("mel", batch)  # torch.tensor
            batch_size_dim, feature_length_dim, time_dim = batch["mel"].shape
            self.assertEqual(batch_size_dim, batch_size)
            self.assertEqual(feature_length_dim, 80)

    def test_dataloaders(self):
        _TOTAL_ITERATIONS = 10
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            dataloaders = get_dataloaders(config_parser)
            for part in ["train", "val"]:
                dl = dataloaders[part]
                for i, batch in tqdm(enumerate(iter(dl)), total=_TOTAL_ITERATIONS,
                                     desc=f"Iterating over {part}"):
                    if i >= _TOTAL_ITERATIONS: break
