"""
    Various functions for generating and analyzing a Plummer Sphere galaxy of equal-mass stars.
"""

from typing import Callable, Any

import numpy as np

rng = np.random.default_rng()


def rng_x(N: int) -> np.ndarray[float, Any]:    
    """Generates a unit Plummer Sphere position distribution.

    Args:
        N (int): Number of points to generate.

    Returns:
        np.ndarray[float, (N,3)]: An N-length array of 3D points.
    """   

    r = (rng.random(N) ** (-2/3) - 1) ** (-1/2)

    return r[np.newaxis].T * unit_sphere(N)


def rng_v(X: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
    """Generates a unit Plummer Sphere velocity distribution based on given positions.

    Args:
        X (np.ndarray[float, (N, 3)]): An N-length array of 3D points.

    Returns:
        np.ndarray[float, (N,3)]: An N-length array of 3D velocities.
    """    

    N = X.shape[0]
    q = montecarlo(N, v_pdf, np.sqrt(2)/3)

    r = np.linalg.norm(X, axis=-1)
    v_escape = 2**(1/2) * (1 + r**2)**(-1/4)

    v = q * v_escape

    return (v)[np.newaxis].T * unit_sphere(N)


def unit_sphere(N: int) -> np.ndarray[float, Any]:
    """Generates N points on the surface of a unit sphere.

    Args:
        N (int): The number of points to generate

    Returns:
        np.ndarray[float, (N,3)]: An N-length array of 3D points.
    """    

    rand = rng.random((N, 2))

    theta = 2 * np.pi * rand[:, 0]

    z = 2 * rand[:, 1] - 1
    s = (1 - z**2) ** 0.5

    x = s * np.cos(theta)
    y = s * np.sin(theta)

    return np.array((x, y, z)).T


def montecarlo(
        N: int,
        pdf: Callable[[float], float],
        y_max: float
        ) -> np.ndarray[float, Any]:
    """Generates numbers according to the given distribution via the Monte Carlo method.

    Args:
        n (int): The number of points to generate.

        pdf (Callable[[float], float]): Probability Density Function (PDF) according to which to generate points.

        y_max (float): Must be greater than or equal to the max value of the PDF. Best results when it's exactly equal (fewer points must be rejected), but will not work if it's less.

    Returns:
        np.ndarray[float, (N,)]: An N-length array of floats distributed according to the PDF.
    """    
    
    result = np.zeros(N)
    for i in range(N):
        while True:
            rand = rng.random(2)
            rand[1] *= y_max

            if rand[1] <= pdf(rand[0]):
                result[i] = rand[0]
                break

    return result


def v_pdf(x: float) -> float:
    """The Plummer Sphere Probability Density Function for velocities. Its max value is sqrt(2/9) ~ 0.471

    Args:
        x (float, [0,1)): The PDF input on the half-open range [0,1).

    Returns:
        float: The PDF value at the point x.
    """    

    return x**2 * (1 - x**2)**(7/2)


def Mass(r: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
    """Calculates the mass contained within r of a Plummer Sphere galaxy of radius R, assuming unit galactic mass (M=1).

    Args:
        r (np.ndarray[float, (N,)]): Radii.

    Returns:
        float: Mass contained within r.
    """    

    return (r)**3 / (1 + r**2)**(3/2)


def K(V: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
    """The pointwise kinetic energy of an array of particles.

    Args:
        V (np.ndarray[float, (N,3)]): An N-length array of 3D velocities.

    Returns:
        np.ndarray[float, (N,)]: An N-length array of kinetic energy magnitudes.
    """    
    return 1/2 * np.linalg.norm(V, axis=-1) ** 2


def U(X: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
    """The pointwise potential energy of a Plummer array of particles.

    Args:
        X (np.ndarray[float, (N,3)]): An N-length array of 3D points.

    Returns:
        np.ndarray[float, (N,)]: An N-length array of potential energy magnitudes.
    """  

    N = X.shape[0]
    r = np.linalg.norm(X, axis=-1)

    return - Mass(r) * (1/N) / r


def E(X: np.ndarray[float, Any], V: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
    """Total pointwise energy of the particles in a Plummer galaxy.

    Args:
        X (np.ndarray[float, (N,3)]): N-length of 3D points.
        V (np.ndarray[float, (N,3)]): N-length array of 3D velocities.

    Returns:
        np.ndarray[float, (N,)]: N-length scalar array of Energies.
    """    
    return U(X) + K(V)