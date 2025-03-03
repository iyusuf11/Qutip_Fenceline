"""Class representing a single pulse."""

import matplotlib.pyplot as plt
import numpy as np


class Pulse:
    """Class representing a single pulse."""

    def __init__(self, omega, amp, phi=0):
        """Initialize the Pulses object with common parameters.

        Args:
            omega (float): Base frequency of the pulses.
            amp (float): Base amplitude of the pulses.
            phi (float): Base phase of the pulses.
        """
        self.omega = omega
        self.amp = amp
        self.phi = phi

    @staticmethod
    def gaussian(t, t0, width, nsig=6):
        """Gaussian pulse shape."""
        return np.exp(-0.5 * ((t - t0 - width * nsig / 2) / width) ** 2)

    @staticmethod
    def smoothbox(t, t0, width, k=0.5, b=3):
        """Smooth box pulse shape."""
        return 0.5 * (np.tanh(k * (t - t0) - b) - np.tanh(k * (t - t0 - width) + b))

    @staticmethod
    def box(t, t0, width):
        """Box pulse shape."""
        return np.heaviside(t - t0, 0) - np.heaviside(t - t0 - width, 0)

    def drive(self, t, args):
        """Drive function applying amplitude and frequency modulation."""
        envelope = args["shape"](t, **args["shape_params"])
        return self.amp * np.cos(self.omega * t + self.phi) * envelope

    @staticmethod
    def plot_pulse(pulses, t_list, show=True):
        """Plot pulse envelopes and their modulated signals on the same plot.

        Parameters:
        - pulses: A list of tuples, each containing a pulse instance and its corresponding args dict.
        - t_list: A list or array of time points at which to evaluate the pulses.
        - show: A boolean indicating whether to show the plot immediately.
        """
        for index, (pulse, args) in enumerate(pulses):
            pulse_shape = args["shape"]
            shape_params = args["shape_params"]
            envelope_values = [pulse_shape(t, **shape_params) for t in t_list]
            modulated_values = [
                pulse.drive(t, {"shape": pulse_shape, "shape_params": shape_params})
                for t in t_list
            ]

            plt.plot(
                t_list,
                modulated_values,
                linestyle="--",
                alpha=0.7,
                label=f"Modulated {index + 1}",
            )
            plt.plot(
                t_list, envelope_values, linewidth=2, label=f"Envelope {index + 1}"
            )

        plt.xlabel("Time (dt)")
        plt.ylabel("Amplitude")
        plt.title("Envelope and Modulated Signals")
        plt.legend()

        if show:
            plt.show()
