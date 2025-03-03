"""Class representing a quantum system."""

from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import qutip as qt
import yaml

from mode import QuantumMode


class QuantumSystem:
    """Class representing a quantum system."""

    def __init__(
        self, modes: List[QuantumMode], couplings: Dict[Tuple[QuantumMode], float]
    ):
        """Initialize a quantum system with given modes and couplings.

        Args:
            modes (list): A list of QuantumMode instances.
            couplings (dict): A dictionary where keys are tuples of coupled QuantumMode instances,
                              and values are the coupling strengths.
        """
        self.modes = modes
        self.couplings = couplings
        self._initialize_mode_operators()

    def _initialize_mode_operators(self):
        """Initialize transformed system ops for each mode in the system.

        Creates and stores annihilators, number operators, and field
        operators for each mode.
        """
        self.modes_a = {mode: self._tensor_op(mode, mode.a) for mode in self.modes}
        self.modes_a_dag = {
            mode: self._tensor_op(mode, mode.a_dag) for mode in self.modes
        }
        self.modes_num = {mode: self._tensor_op(mode, mode.num) for mode in self.modes}
        self.modes_field = {
            mode: self._tensor_op(mode, mode.field) for mode in self.modes
        }
        self.modes_Z = {mode: self._tensor_op(mode, mode.Z) for mode in self.modes}

    def _tensor_op(self, mode: QuantumMode, op: qt.Qobj):
        """Tensor an operator with the identity operator on all other modes.

        Args:
            mode (QuantumMode): The mode for which the operator is defined.
            op (qutip.Qobj): The operator to be tensored.

        Returns:
            qutip.Qobj: The tensored operator in the Hilbert space of the entire system.
        """
        mode_index = self.modes.index(mode)
        op_list = [qt.qeye(m.dim) for m in self.modes]
        op_list[mode_index] = op
        return reduce(qt.tensor, op_list)

    def prepare_tensor_fock_state(self, mode_states: List[Tuple[QuantumMode, int]]):
        """Prepare a Fock product state for specified modes.

        Args:
            mode_states (list of tuples): Each tuple contains a QuantumMode object and an integer
                                          representing the Fock state number for that mode.
                                          Modes not included in the list are assumed to be in the 0 state.

        Returns:
            qutip.Qobj: Tensor product state as a QuTiP object.
        """
        state_list = [qt.basis(mode.dim, 0) for mode in self.modes]
        for mode, state in mode_states:
            if mode not in self.modes:
                raise ValueError(f"Mode {mode} not found in system.")
            state_list[self.modes.index(mode)] = qt.basis(mode.dim, state)

        return reduce(qt.tensor, state_list)

    def mode_population_expectation(
        self, system_state: qt.Qobj, mode: QuantumMode, fock_state: int
    ):
        """Calculate expectation of the population of a mode in a given state.

        Args:
            system_state (qutip.Qobj): The state of the entire quantum system.
            mode (QuantumMode): The mode for which the population is calculated.
            fock_state (int): The Fock state number for the mode.

        Returns:
            float: The expectation value of the mode population.
        """
        fock_state = qt.basis(mode.dim, fock_state)
        fock_state_op = fock_state * fock_state.dag()
        system_fock_state_op = self._tensor_op(mode, fock_state_op)
        return qt.expect(system_fock_state_op, system_state)

    def __repr__(self) -> str:
        """Return a string representation of the QuantumSystem."""
        return f"QuantumSystem({self.modes})"

    @classmethod
    def from_yaml(cls, file_path: str):
        """Create a QuantumSystem instance from a YAML file.

        Args:
            file_path (str): Path to the YAML file containing modes and couplings data.

        Returns:
            QuantumSystem: An instance of QuantumSystem initialized from the YAML file.
        """
        yaml_path = Path(file_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found at: {yaml_path}")

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        modes = []
        for name, properties in data["modes"].items():
            mode = QuantumMode(name=name, **properties)
            modes.append(mode)

        # Process couplings
        couplings = {}
        for _, coupling in data["couplings"].items():
            mode_names = coupling["modes"]
            mode_objs = [
                next(mode for mode in modes if mode.name == name) for name in mode_names
            ]

            # Convert g2 from GHz to rad/s
            couplings[tuple(mode_objs)] = coupling["g2"] * 2 * np.pi

        return cls(modes, couplings)
    def get_mode(self, name: str):
        """Retrieve a QuantumMode by name.

        Args:
            name (str): The name of the mode.

        Returns:
            QuantumMode: The corresponding QuantumMode object.

        Raises:
            ValueError: If the mode is not found.
        """
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise ValueError(f"Mode '{name}' not found in the system.")