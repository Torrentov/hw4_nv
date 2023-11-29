import torch
import torch.nn.functional as F

from hw_nv.preprocessing.mel_spectrogram import MelSpectrogram

import logging
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    wave2spec = MelSpectrogram()

    audio_max_length = max(item['audio'].size(1) for item in dataset_items)

    result_batch['audio_gt'] = []
    result_batch['audio_path'] = []
    spectrograms = []

    for elem in dataset_items:
        wave_current_length = elem['audio'].shape[1]
        wave_padding_needed = audio_max_length - wave_current_length
        padding_tensor = torch.zeros(1, wave_padding_needed)
        wave_padded = torch.cat((elem['audio'], padding_tensor), dim=1)
        result_batch['audio_gt'].append(wave_padded)
        result_batch['audio_path'].append(elem['audio_path'])
        with torch.no_grad():
            audio_spectrogram = wave2spec(wave_padded)
        spectrograms.append(audio_spectrogram)

    result_batch['audio_gt'] = torch.stack(result_batch['audio_gt'])

    max_time = max(spec.shape[-1] for spec in spectrograms)
    padded_spectrograms = [F.pad(spec, (0, max_time - spec.shape[-1])) for spec in spectrograms]
    result_batch["mel"] = torch.stack(padded_spectrograms).squeeze(1)
    return result_batch
