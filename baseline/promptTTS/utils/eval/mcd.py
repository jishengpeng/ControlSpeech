from typing import Tuple

import numpy as np
from fastdtw import fastdtw
from librosa.feature import mfcc
from scipy.spatial.distance import euclidean


Penalty = float
MelCepstralDistance = float
Frames = int

def get_mfccs_of_mel_spectogram(mel_spectogram: np.ndarray, n_mfcc: int, take_log: bool) -> np.ndarray:
  mel_spectogram = np.log10(mel_spectogram) if take_log else mel_spectogram
  mfccs = mfcc(
    S=mel_spectogram,
    n_mfcc=n_mfcc + 1,
    norm=None,
    y=None,
    sr=None,
    dct_type=2,
    lifter=0,
  )

  # according to "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek, the zeroth
  # coefficient is omitted
  # there are different variants of the Discrete Cosine Transform Type II, the one that librosa's MFCC uses is 2 times
  # bigger than the one we want to use (which appears in Kubicheks paper)
  mfccs = mfccs[1:] / 2

  return mfccs


def get_mcd_and_penalty_and_final_frame_number(mfccs_1: np.ndarray, mfccs_2: np.ndarray, use_dtw: bool
                                               ) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  former_frame_number_1 = mfccs_1.shape[1]
  former_frame_number_2 = mfccs_2.shape[1]
  mcd, final_frame_number = equal_frame_numbers_and_get_mcd(
    mfccs_1, mfccs_2, use_dtw)
  penalty = get_penalty(former_frame_number_1,
                        former_frame_number_2, final_frame_number)
  return mcd, penalty, final_frame_number


def equal_frame_numbers_and_get_mcd(mfccs_1: np.ndarray, mfccs_2: np.ndarray,
                                    use_dtw: bool) -> Tuple[MelCepstralDistance, Frames]:
  if mfccs_1.shape[0] != mfccs_2.shape[0]:
    raise Exception("The number of coefficients per frame has to be the same for both inputs.")
  if use_dtw:
    mcd, final_frame_number = get_mcd_with_dtw(mfccs_1, mfccs_2)
    return mcd, final_frame_number
  mfccs_1, mfccs_2 = fill_rest_with_zeros(mfccs_1, mfccs_2)
  mcd, final_frame_number = get_mcd(mfccs_1, mfccs_2)
  return mcd, final_frame_number


def fill_rest_with_zeros(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  frame_number_1 = mfccs_1.shape[1]
  frame_number_2 = mfccs_2.shape[1]
  diff = abs(frame_number_1 - frame_number_2)
  if diff > 0:
    adding_array = np.zeros(shape=(mfccs_1.shape[0], diff))
    if frame_number_1 < frame_number_2:
      mfccs_1 = np.concatenate((mfccs_1, adding_array), axis=1)
    else:
      mfccs_2 = np.concatenate((mfccs_2, adding_array), axis=1)
  assert mfccs_1.shape == mfccs_2.shape
  return mfccs_1, mfccs_2


def get_mcd_with_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[MelCepstralDistance, Frames]:
  mfccs_1, mfccs_2 = mfccs_1.T, mfccs_2.T
  distance, path = fastdtw(mfccs_1, mfccs_2, dist=euclidean)

  _, path = fastdtw(mfccs_1, mfccs_2, dist=euclidean)
  twf = np.array(path).T
  mfccs_1_dtw = mfccs_1[twf[0]]
  mfccs_2_dtw = mfccs_2[twf[1]]

  # MCD
  diff2sum = np.sum((mfccs_1_dtw - mfccs_2_dtw) ** 2, 1)
  mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

  final_frame_number = len(path)
  mcd = distance / final_frame_number
  return mcd, final_frame_number


def get_mcd(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[MelCepstralDistance, Frames]:
  assert mfccs_1.shape == mfccs_2.shape
  mfccs_diff = mfccs_1 - mfccs_2
  mfccs_diff_norms = np.linalg.norm(mfccs_diff, axis=0)
  mcd = np.mean(mfccs_diff_norms)
  frame_number = len(mfccs_diff_norms)
  return mcd, frame_number


def get_penalty(former_length_1: int, former_length_2: int, length_after_equaling: int) -> Penalty:
  # lies between 0 and 1, the smaller the better
  penalty = 2 - (former_length_1 + former_length_2) / length_after_equaling
  return penalty

def get_metrics_mels(mel_1: np.ndarray, mel_2: np.ndarray, *, n_mfcc: int = 16, take_log: bool = False, use_dtw: bool = True) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.
    Parameters
    ----------
    mel_1 	  : np.ndarray [shape=(k,n)]
        first mel spectogram
    mel_2     : np.ndarray [shape=(k,m)]
        second mel spectogram
    take_log     : bool
        should be set to `False` if log10 already has been applied to the input mel spectograms, otherwise `True`
    n_mfcc    : int > 0 [scalar]
        the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
        zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
        according to Robert F. Kubichek)
    use_dtw:  : bool [scalar]
        to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
        `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
        otherwise the array with less columns is filled with zeros from the right side.
    Returns
    -------
    mcd         : float
        the mel-cepstral distance between the two input audios
    penalty     : float
        a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
        with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
        coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
        each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
        reached if no columns were added to either array.
    final_frame_number : int
        the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
        filling up with zeros
    """

  if mel_1.shape[0] != mel_2.shape[0]:
    raise ValueError(
      "The amount of mel-bands that were used to compute the corresponding mel-spectogram have to be equal!")
  mfccs_1 = get_mfccs_of_mel_spectogram(mel_1, n_mfcc, take_log)
  mfccs_2 = get_mfccs_of_mel_spectogram(mel_2, n_mfcc, take_log)
  mcd, penalty, final_frame_number = get_mcd_and_penalty_and_final_frame_number(
    mfccs_1=mfccs_1, mfccs_2=mfccs_2, use_dtw=use_dtw)
  return mcd, penalty, final_frame_number