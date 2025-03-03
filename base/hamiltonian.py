"""Hamiltonian for a quantum system."""

# FIXME after making some changes to the code -
# I am unsure if the docstring equations are all accurate


from abc import ABC

import numpy as np

from mode import CavityMode, QuantumMode, SNAILMode
from pulse import Pulse
from system import QuantumSystem


class Hamiltonian(ABC):
    """Hamiltonian for a quantum system."""

    def __init__(self, quantum_system: QuantumSystem, use_RWA=True, use_TLS=True):
        """Initialize the Hamiltonian for a given quantum system.

        Args:
            quantum_system (QuantumSystem): The quantum system for which the Hamiltonian is constructed.
            use_RWA (bool): Flag to indicate whether to use the Rotating Wave Approximation.
            use_TLS (bool): Flag to indicate whether to use the Two-Level System approximation.
        """
        self.system = quantum_system
        self.use_TLS = use_TLS
        self.use_RWA = use_RWA

    def _build_H(self):
        """Generate the Hamiltonian for the system."""
        self.H = 0

        for mode in self.system.modes:
            self.H += mode.H_0(self.system, RWA=self.use_RWA, TLS=self.use_TLS)

        self.H += self._H_int()

    def _H_int(self):
        r"""Generate the coupling part of the Hamiltonian.

        This includes terms for each coupling in the system:
        :math:`H_{\text{int}} = 2 \pi \sum_{\text{couplings}} g_2 (\hat{a}_m + \hat{a}_m^{\dagger})(\hat{a}_n + \hat{a}_n^{\dagger})`
        where \( g_2 \) is the coupling strength between modes \( m \) and \( n \).
        """
        H_int = 0
        for (mode1, mode2), g2 in self.system.couplings.items():
            _field1 = self.system.modes_field[mode1]
            _field2 = self.system.modes_field[mode2]
            H_int += g2 * _field1 * _field2

        return H_int


class QubitQubitSNAIL(Hamiltonian):
    """Hamiltonian for a quantum system with two qubits and a SNAIL mode."""

    def __init__(self, quantum_system: QuantumSystem, use_RWA=True, use_TLS=True):
        """Initialize the Hamiltonian for a given quantum system."""
        super().__init__(quantum_system, use_RWA, use_TLS)

        # grab references to modes
        self.q1_mode = self.system.modes[0]  # Q1
        self.q2_mode = self.system.modes[1]  # Q2
        self.snail_mode = self.system.modes[2]  # SNAIL
        self.snail_field = self.system.modes_field[self.snail_mode]

        # make sure our quantum system is set up correctly
        assert isinstance(self.q1_mode, QuantumMode)
        assert isinstance(self.q2_mode, QuantumMode)
        assert isinstance(self.snail_mode, SNAILMode)
        assert len(self.system.modes) == 3

        # build the Hamiltonian, using standard coupling terms
        self._build_H()

    def driven(self, pulse: Pulse):
        """Return the Hamiltonian with the pulse applied."""
        return [self.H, [self.snail_field, pulse.drive]]
    

class QubitQubitQubitSNAIL(Hamiltonian):
    """Hamiltonian for a quantum system with two qubits and a SNAIL mode."""

    def __init__(self, quantum_system: QuantumSystem, use_RWA=True, use_TLS=True):
        """Initialize the Hamiltonian for a given quantum system."""
        super().__init__(quantum_system, use_RWA, use_TLS)

        # grab references to modes
        self.q1_mode = self.system.modes[0]  # Q1
        self.q2_mode = self.system.modes[1]  # Q2
        self.q3_mode = self.system.modes[2]  # Q3
        self.snail_mode = self.system.modes[3]  # SNAIL
        self.snail_field = self.system.modes_field[self.snail_mode]

        # make sure our quantum system is set up correctly
        assert isinstance(self.q1_mode, QuantumMode)
        assert isinstance(self.q2_mode, QuantumMode)
        assert isinstance(self.q3_mode, QuantumMode)
        assert isinstance(self.snail_mode, SNAILMode)
        assert len(self.system.modes) == 4

        # build the Hamiltonian, using standard coupling terms
        self._build_H()

    def driven(self, pulse: Pulse):
        """Return the Hamiltonian with the pulse applied."""
        return [self.H, [self.snail_field, pulse.drive]]


class QubitQubitCavity(Hamiltonian):
    """Hamiltonian for a quantum system with two qubits and a cavity mode.

    Reference:
    [1] A. Blais, R.-S. Huang, A. Wallraff, S. M. Girvin, and R. J. Schoelkopf
    doi: 10.1103/PhysRevA.69.062320.
    [2] A. Blais et al., Phys. Rev. A, vol. 75, no. 3, p. 032329, Mar. 2007, doi: 10.1103/PhysRevA.75.032329.
    """

    def __init__(self, quantum_system: QuantumSystem):
        """Initialize the Hamiltonian for a given quantum system."""
        super().__init__(quantum_system)

        # grab references to modes
        self.q1_mode = self.system.modes[0]  # Q1
        self.q2_mode = self.system.modes[1]  # Q2
        self.cavity_mode = self.system.modes[2]

        # make sure our quantum system is set up correctly
        assert isinstance(self.q1_mode, QuantumMode)
        assert isinstance(self.q2_mode, QuantumMode)
        assert isinstance(self.cavity_mode, CavityMode)
        assert len(self.system.modes) == 3

        self._build_H()

    def _build_H(self):
        """Generate the Hamiltonian for the system."""
        self.H = 0

        for mode in self.system.modes:
            self.H += mode.H_0(self.system, RWA=self.use_RWA, TLS=self.use_TLS)

    # override the interaction term
    def _H_int(self):
        r"""Generate the coupling part of the Hamiltonian.

        Equation [1](32), substracting the non-interacting terms which
        will be included by self._build_H()

        Returns a list of terms [H1, H2, H3] where H1 is scaled by
        \eta1(t) H2 is scaled by \eta2(t) H3 is scaled by \eta1(t) *
        \eta2(t)
        """
        H1, H2, H3 = 0, 0, 0

        _g1 = self.system.couplings[(self.q1_mode, self.cavity_mode)]
        _g2 = self.system.couplings[(self.q2_mode, self.cavity_mode)]

        _delta1 = np.abs(self.q1_mode.freq - self.cavity_mode.freq)
        _delta2 = np.abs(self.q2_mode.freq - self.cavity_mode.freq)

        # get reference to operators relative to composite quantum system
        _q1_a = self.system.modes_a[self.q1_mode]
        _q1_adag = self.system.modes_a_dag[self.q1_mode]
        _q2_a = self.system.modes_a[self.q2_mode]
        _q2_adag = self.system.modes_a_dag[self.q2_mode]
        _q1_Z = self.system.modes_Z[self.q1_mode]
        _q2_Z = self.system.modes_Z[self.q2_mode]
        _cavity_num = self.system.modes_num[self.cavity_mode]

        H1 = _g1**2 / _delta1 * _q1_Z * _cavity_num
        H1 += _g1**2 / (2 * _delta1) * _q1_Z

        H2 = _g2**2 / _delta2 * _q2_Z * _cavity_num
        H2 += _g2**2 / (2 * _delta2) * _q2_Z

        # _J = _g1 * _g2 * (_delta1 + _delta2) / (2 * _delta1 * _delta2)
        _J = _g1 * _g2 * (1 / _delta1 + 1 / _delta2) / 2
        H3 = _J * (_q1_a * _q2_adag + _q1_adag * _q2_a)

        return [H1, H2, H3]

    def driven(self, pulse1: Pulse, pulse2: Pulse):
        r"""Return the Hamiltonian with the two pulses applied.

        Args:
            pulse1 (Pulse): The first pulse to apply.
            pulse2 (Pulse): The second pulse to apply.

        These pulses define \eta(t), some time-dependent function that scales the coupling terms, g2.
        When eta(t) = 0, the coupling terms are unchanged from the inherent physical system.
        We define two time-dependent functions, which control the couplings between qubit1 and the cavity, and
        between qubit2 and the cavity, respectively.

        Given the Hamiltonian [1](32), [2](5.2), we then factor out the time-dependent functions. In order for these to work
        with mesolve(), they must be linearly multiplied by the time-dependent pulse functions. Therefore, from the Hamiltonian
        we end up factoring (1+2\eta+\eta^2) from the first coupling term, (1+2\eta+\eta^2) from the second coupling term,
        and (1+\eta1+\eta2+\eta1\eta2) from the third coupling term. These are the functions returned by this method.
        """
        _H1, _H2, _H3 = self._H_int()

        def transformed_pulse1(t, args):
            # eta = pulse1.drive(t, args)
            eta = pulse1(t, args)
            return 1 + 2 * eta + eta**2

        def transformed_pulse2(t, args):
            # eta = pulse2.drive(t, args)
            eta = pulse2(t, args)
            return 1 + 2 * eta + eta**2

        def transformed_pulse3(t, args):
            # eta1 = pulse1.drive(t, args)
            eta1 = pulse1(t, args)
            # eta2 = pulse2.drive(t, args)
            eta2 = pulse2(t, args)
            return 1 + eta1 + eta2 + eta1 * eta2

        return [
            self.H,
            [_H1, transformed_pulse1],
            [_H2, transformed_pulse2],
            [_H3, transformed_pulse3],
        ]
