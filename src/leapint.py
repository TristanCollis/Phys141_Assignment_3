from typing import Any
import numpy as np

from . import gravnb as grav


def integrate(x_init: np.ndarray[float, Any], v_init: np.ndarray[float, Any], masses: np.ndarray[float, Any], steps: int, dt: float, epsilon: float) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
    x_history = np.zeros((steps, *x_init.shape))
    x_history[0] = x_init

    v_history = np.zeros_like(x_history)
    v_history[0] = v_init

    acceleration = grav.grav_direct(x_init, masses, epsilon)

    for i in range(steps-1):
        v_halfstep = v_history[i] + acceleration * dt/2

        x_history[i+1] = x_history[i] + v_halfstep * dt

        acceleration = grav.grav_direct(x_history[i+1], masses, epsilon)

        v_history[i+1] = v_halfstep + acceleration * dt/2

    return x_history, v_history
