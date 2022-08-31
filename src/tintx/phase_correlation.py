"""
tint.phase_correlation
======================

Functions for performing phase correlation. Used to predict cell movement
between scans.

"""
from __future__ import annotations
from typing import Optional, overload, Union
from typing_extensions import Literal

import numpy as np
from scipy import ndimage

from .types import ConfigType


def get_ambient_flow(
    obj_extent: dict[str, np.ndarray],
    img1: np.ndarray,
    img2: np.ndarray,
    params: ConfigType,
    grid_size: Union[np.ndarray, tuple[int, int, int]],
) -> Optional[float]:
    """Takes in object extent and two images and returns ambient flow. Margin
    is the additional region around the object used to compute the flow
    vectors."""
    margin_r = params["FLOW_MARGIN"] / grid_size[1]
    margin_c = params["FLOW_MARGIN"] / grid_size[2]
    row_lb = obj_extent["obj_center"][0] - obj_extent["obj_radius"] - margin_r
    row_ub = obj_extent["obj_center"][0] + obj_extent["obj_radius"] + margin_r
    col_lb = obj_extent["obj_center"][1] - obj_extent["obj_radius"] - margin_c
    col_ub = obj_extent["obj_center"][1] + obj_extent["obj_radius"] + margin_c
    row_lb = int(row_lb)
    row_ub = int(row_ub)
    col_lb = int(col_lb)
    col_ub = int(col_ub)

    dims = img1.shape

    row_lb = np.max([row_lb, 0])
    row_ub = np.min([row_ub, dims[0]])
    col_lb = np.max([col_lb, 0])
    col_ub = np.max([col_ub, dims[1]])

    flow_region1 = np.copy(img1[row_lb : row_ub + 1, col_lb : col_ub + 1])
    flow_region2 = np.copy(img2[row_lb : row_ub + 1, col_lb : col_ub + 1])

    flow_region1[flow_region1 != 0] = 1
    flow_region2[flow_region2 != 0] = 1
    return fft_flowvectors(flow_region1, flow_region2)


def fft_flowvectors(
    im1: np.ndarray, im2: np.ndarray, global_shift: bool = False
) -> Optional[float]:
    """Estimates flow vectors in two images using cross covariance."""
    if not global_shift and (np.max(im1) == 0 or np.max(im2) == 0):
        return None

    crosscov = fft_crosscov(im1, im2)
    sigma = (1 / 8) * min(crosscov.shape)
    cov_smooth = ndimage.gaussian_filter(crosscov, sigma)
    dims = np.array(im1.shape)
    pshift = np.argwhere(cov_smooth == np.max(cov_smooth))[0]
    pshift = (pshift + 1) - np.round(dims / 2, 0)
    return pshift


def fft_crosscov(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """Computes cross correlation matrix using FFT method."""
    fft1_conj = np.conj(np.fft.fft2(im1))
    fft2 = np.fft.fft2(im2)
    normalize = abs(fft2 * fft1_conj)
    normalize[normalize == 0] = 1  # prevent divide by zero error
    cross_power_spectrum = (fft2 * fft1_conj) / normalize
    crosscov = np.fft.ifft2(cross_power_spectrum)
    crosscov = np.real(crosscov)
    return fft_shift(crosscov)


def fft_shift(fft_mat: np.ndarray) -> np.ndarray:
    """Rearranges the cross correlation matrix so that 'zero' frequency or DC
    component is in the middle of the matrix. Taken from stackoverflow Que.
    30630632."""
    rd2 = int(fft_mat.shape[0] / 2)
    cd2 = int(fft_mat.shape[1] / 2)
    quad1 = fft_mat[:rd2, :cd2]
    quad2 = fft_mat[:rd2, cd2:]
    quad3 = fft_mat[rd2:, cd2:]
    quad4 = fft_mat[rd2:, :cd2]
    centered_t = np.concatenate((quad4, quad1), axis=0)
    centered_b = np.concatenate((quad3, quad2), axis=0)
    centered = np.concatenate((centered_b, centered_t), axis=1)
    return centered


@overload
def get_global_shift(im1: Literal[None], im2: Literal[None]) -> None:
    ...  # pragma: no cover


@overload
def get_global_shift(im1: np.ndarray, im2: Literal[None]) -> None:
    ...  # pragma: no cover


@overload
def get_global_shift(im1: Literal[None], im2: np.ndarray) -> None:
    ...  # pragma: no cover


@overload
def get_global_shift(im1: np.ndarray, im2: np.ndarray) -> float:
    ...  # pragma: no cover


def get_global_shift(
    im1: Optional[np.ndarray], im2: Optional[np.ndarray]
) -> Optional[float]:
    """Returns standardazied global shift vector. im1 and im2 are full frames
    of raw DBZ values."""
    if im2 is None or im1 is None:
        return None
    shift = fft_flowvectors(im1, im2, global_shift=True)
    return shift
