"""Class representing a single mode in a quantum system."""

from abc import ABC, abstractmethod

import numpy as np
import qutip as qt


class QuantumMode(ABC):
    """Class representing a single mode in a quantum system."""

    def __new__(cls, **kwargs):
        """Create a new QuantumMode instance of the appropriate subclass."""
        mode_type = kwargs["mode_type"]
        if cls is QuantumMode:
            if mode_type == "Qubit":
                return super().__new__(QubitMode)
            elif mode_type == "Cavity":
                return super().__new__(CavityMode)
            elif mode_type == "SNAIL":
                return super().__new__(SNAILMode)
            else:
                raise ValueError(f"Unknown mode type: {mode_type}")
        return super().__new__(cls)

    def __init__(self, **kwargs):
        """Initialize a QuantumMode instance representing a single mode.

        Args:
            name (str): The name of the quantum mode.
            freq (float): The frequency of the quantum mode (GHz).
            dim (int): The dimension of the Hilbert space for this mode.
        """
        self.name = kwargs["name"]
        self.freq = kwargs["freq"] * 2 * np.pi  # Convert from GHz to rad/ns
        self.dim = kwargs["dim"]

        # lifetimes
        self.T1 = kwargs.get("T1", 100000)  # relaxation time (ns)
        self.T2 = kwargs.get("T2", 50000)  # dephasing time (ns)
        self.Tphi = 1 / (1 / self.T2 - 1 / (2 * self.T1))  # phase coherence time (ns)

        # collapse rates
        self.gamma_1 = 1 / self.T1 if self.T1 > 0 else 0
        self.gamma_phi = 1 / self.Tphi if self.Tphi > 0 else 0

        # Initialize quantum operators
        self.a = qt.destroy(self.dim)  # Annihilation operator
        self.a_dag = self.a.dag()  # Creation operator
        self.num = qt.num(self.dim)  # Number operator
        self.field = self.a + self.a_dag  # Field operator
        if self.dim == 2:
            self.Z = qt.sigmaz()
        # FIXME, not a well defined operator
        else:
            self.Z = qt.identity(self.dim)
        # self.Z = self.a * self.a_dag - self.a_dag * self.a  # Z operator

    def __repr__(self) -> str:
        """Return a string representation of the QuantumMode."""
        return f"{self.__class__.__name__}(name={self.name}, freq={self.freq / (2 * np.pi)} GHz, dim={self.dim})"

    def _get_operators(self, system):
        """Return the appropriate set of operators depending on the context."""
        if system is not None:
            return (
                system.modes_a[self],
                system.modes_a_dag[self],
                system.modes_num[self],
                system.modes_field[self],
                system.modes_Z[self],
            )
        return self.a, self.a_dag, self.num, self.field, self.Z

    def collapse_operators(self, system=None):
        """Return the collapse operators for the mode.

        See qutip.lindblad_dissipator and qutip.

        Args:
            system (QuantumSystem, optional): The quantum system to which the mode belongs.

        Returns:
            list: The collapse operators for the mode.
        """
        a, _, num, _, _ = self._get_operators(system)
        return [np.sqrt(self.gamma_1) * a, np.sqrt(self.gamma_phi / 2) * num]

    @abstractmethod
    def H_0(self, system=None, **kwargs):
        """Calculate the non-perturbed Hamiltonian for the mode.

        Args:
            system (QuantumSystem, optional): The quantum system to which the mode belongs.
                If provided, the Hamiltonian is calculated using operators that are part of the
                larger system's Hilbert space. If None, the Hamiltonian is calculated for the mode in isolation.

        Returns:
            qutip.Qobj: The Hamiltonian operator for this mode.
        """
        pass


class QubitMode(QuantumMode):
    """Class representing a single Qubit mode in a quantum system."""

    def __init__(self, **kwargs):
        """Initialize a QubitMode instance representing a single Qubit mode."""
        super().__init__(**kwargs)
        self.alpha = kwargs["alpha"] * 2 * np.pi  # Convert alpha from GHz to rad/s

    def H_0(self, system=None, **kwargs):
        """Calculate the Hamiltonian for a Qubit mode, with optional RWA/TLS.

        Args:
            system (QuantumSystem, optional): The quantum system to which the mode belongs.
            RWA (bool, optional): Flag to use the Rotating Wave Approximation. Defaults to True.
            TLS (bool, optional): Flag to use the Two-Level System approximation. Defaults to True.

        Returns:
            qutip.Qobj: The Hamiltonian operator for this Qubit mode.
        """
        RWA = kwargs.get("RWA", True)
        TLS = kwargs.get("TLS", True)

        a, a_dag, num, _, Z = self._get_operators(system)

        if TLS:  # TLS overrides RWA
            if self.dim != 2:
                raise ValueError("TLS approximation requires a 2-level system.")
            return -self.freq * Z / 2
        elif RWA:
            alpha_term = self.alpha / 2 * a_dag * a_dag * a * a
            return self.freq * num + alpha_term
        else:
            return self.freq * num  + self.alpha/2 * (a + a_dag)**4
            # return (self.freq - self.alpha) * num + self.alpha / 12 * num**2
            # return (self.freq - self.alpha) * num + self.alpha / 12 * num**4


class CavityMode(QuantumMode):
    """Class representing a single Cavity mode in a quantum system."""

    def __init__(self, **kwargs):
        """Initialize a CavityMode instance."""
        super().__init__(**kwargs)

    def H_0(self, system=None, **kwargs):
        """Calculate the Hamiltonian for a Cavity mode.

        Args:
            system (QuantumSystem, optional): The quantum system to which the mode belongs.

        Returns:
            qutip.Qobj: The Hamiltonian operator for this Cavity mode.
        """
        _, _, num, _, _ = self._get_operators(system)
        return self.freq * num


class SNAILMode(QuantumMode):
    """Class representing a single SNAIL mode in a quantum system."""

    def __init__(self, **kwargs):
        """Initialize a SNAILMode instance representing a single SNAIL mode."""
        super().__init__(**kwargs)
        self.g3 = kwargs["g3"] * 2 * np.pi  # Convert g3 from GHz to rad/s

    def H_0(self, system=None, **kwargs):
        """Calculate the Hamiltonian for a SNAIL mode, with optional RWA.

        Args:
            system (QuantumSystem, optional): The quantum system to which the mode belongs.
            RWA (bool, optional): Flag to use the Rotating Wave Approximation. Defaults to True.

        Returns:
            qutip.Qobj: The Hamiltonian operator for this SNAIL mode.
        """
        RWA = kwargs.get("RWA", True)

        a, a_dag, num, field, _ = self._get_operators(system)

        # for some reason, SNAIL RWA breaks my simulation

        if RWA:
            g3_term = self.g3 / 2 * (a_dag * a * a + a * a_dag * a_dag)
            return self.freq * num + g3_term
        else:
            return self.freq * num + self.g3 / 6 * field**3
