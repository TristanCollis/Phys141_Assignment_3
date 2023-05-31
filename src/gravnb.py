from typing import Any

import numpy as np
import numba as nb  # type: ignore
from numba import jit, njit, float64, int32


@njit(["float64[:, :, :](float64[:, :])", "float64[:,:](float64[:])"])
def add_dim(array: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
    output = np.empty((1, *array.shape))
    output[0] = array

    return output


@njit(["float64[:,:](float64[:,:,:])", "float64[:](float64[:,:])"])
def norm(array: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
    return np.sqrt(
        np.sum(
            array**2,
            axis=-1
        )
    )


@njit(float64[:,:](float64[:,:], float64[:,:], float64[:], float64),)
def grav_3body(
        massless_points: np.ndarray[float, Any],
        massive_points: np.ndarray[float, Any],
        masses: np.ndarray[float, Any],
        epsilon: float,
) -> np.ndarray[float, Any]:
    """Restricted N-Body acceleration using a small number of massive bodies attracting a large number of massless bodies.

    Parameters
    ----------
    massless_points : np.ndarray[float, (m, 3)]
        Array of massless points.
    massive_points : np.ndarray[float, (n, 3)]
        Array of massive points.
    masses : np.ndarray[float, (n)]
        Masses of massive points.
    epsilon : float
        Softening factor

    Returns
    -------
    np.ndarray[float, (m, 3)]
        Accelerations to be applied to the massless bodies.
    """

    r = add_dim(massive_points).transpose(1, 0, 2) - add_dim(massless_points)

    r_norm = add_dim(norm(r)).transpose(1, 2, 0)

    return np.sum(
        add_dim((add_dim(masses))).T
        * r
        / (r_norm + epsilon)**3,
        axis=0
    )

@njit(float64[:,:](float64[:,:], float64[:], float64))
def grav_direct(
        points: np.ndarray[float, Any],
        masses: np.ndarray[float, Any],
        epsilon: float,
) -> np.ndarray[float, Any]:
    """Direct gravity acceleration between n massive points.

    Parameters
    ----------
    points : np.ndarray[float, (n, 3)]
        Positions of the massive points.
    masses : np.ndarray[float, (n)]
        Masses corresponding to the given points.
    epsilon : float
        Softening factor

    Returns
    -------
    np.ndarray[float, (n, 3)]
        Acceleration on each massive point.
    """

    return grav_3body(points, points, masses, epsilon)
