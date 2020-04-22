#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, MutableMapping, Optional
from pathlib import Path
from pickle import dump

import numpy as np
import torch
from torch.utils.data import DataLoader
from librosa.core import load as lb_load, stft
from librosa.filters import mel

__docformat__ = 'reStructuredText'
__all__ = [
    'get_audio_file_data',
    'extract_mel_band_energies',
    'serialize_features_and_classes',
    'serialize_features_and_classes',
    'dataset_iteration'
]


def get_audio_file_data(audio_file: str) \
        -> np.ndarray:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    return lb_load(path=audio_file, sr=None, mono=True)[0]


def extract_mel_band_energies(audio_file: str,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              hop_length: Optional[int] = 512,
                              n_mels: Optional[int] = 40) \
        -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 512.
    :type hop_length: Optional[int]
    :param n_mels: Number of MEL frequencies/filters to be used, defaults to 40.
    :type n_mels: Optional[int]
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    spec = stft(
        y=get_audio_file_data(audio_file=audio_file),
        n_fft=n_fft,
        hop_length=hop_length)

    mel_filters = mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    return np.dot(mel_filters, np.abs(spec) ** 2)


def serialize_features_and_classes(
        f_name: Path,
        features_and_classes: MutableMapping[str, Union[np.ndarray, int]],
        output_directory: Optional[Union[Path, None]] = None) \
        -> None:
    """Serializes the features and classes.

    :param f_name: File name of the output file.
    :type f_name: Path
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    :param output_directory: Output directory for the features and classes, defaults to None.
    :type output_directory: Optional[Path|None]
    """
    f_path = f_name if output_directory is None else output_directory.joinpath(f_name)
    with f_path.open('rb') as f:
        dump(features_and_classes, f)


def dataset_iteration(dataset: torch.utils.data.Dataset,
                      batch_size: Optional[int] = 1,
                      shuffle: Optional[bool] = True) \
        -> None:
    """Iterates over the dataset using the DataLoader of PyTorch.

    :param dataset: Dataset to iterate over.
    :type dataset: torch.utils.data.Dataset
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param shuffle: Shall we shuffle the examples?
    :type shuffle: bool
    """
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)

    for data in data_loader:
        x, y = data
        print(f'x type: {type(x)} | y type: {type(y)}')
        print(f'x size: {x.size()} | y size: {y.size()}')

# EOF
