import numpy as np
from qutip import Options
import qutip as qt
from functools import reduce
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
# from system_params import modes, coupling

PI = np.pi
opts = qt.Options(nsteps=1e6, atol=1e-8, rtol=1e-6)
# opts = qt.Options(nsteps=1e6, atol=1e-14, rtol=1e-12)
p_bar = True  # qt.ui.TextProgressBar()

class QuantumMode:
    def __init__(self, name: str, freq: float, dim: int, **kwargs):
        """
        Initialize a QuantumMode instance representing a single mode in a quantum system.

        Args:
            name (str): The name of the quantum mode.
            freq (float): The frequency of the quantum mode.
            dim (int): The dimension of the Hilbert space for this mode.
            **kwargs: Additional properties of the mode, e.g., 'alpha' for qubits, 'g3' for SNAILs.

        The kwargs are used to add any additional attributes specific to different types of quantum modes.
        """
        self.name = name
        self.freq = freq
        self.dim = dim
        self.__dict__.update(kwargs)

        # Initialize quantum operators
        self.a = qt.destroy(dim)  # Annihilation operator
        self.a_dag = self.a.dag()  # Creation operator
        self.num = qt.num(dim)  # Number operator
        self.field = self.a + self.a_dag  # Field operator

        # verify has attribute g3 or alpha, but not both
        assert hasattr(self, "g3") ^ hasattr(self, "alpha")

    def __repr__(self) -> str:
        return f"QuantumMode(name={self.name}, freq={self.freq} GHz, dim={self.dim})"

class QuantumSystem:
    def __init__(
        self, modes: List[QuantumMode], couplings: Dict[Tuple[QuantumMode], float]
    ):
        """
        Initialize a quantum system with given modes and couplings.

        Args:
            modes (list): A list of QuantumMode instances.
            couplings (dict): A dictionary where keys are tuples of coupled QuantumMode instances,
                              and values are the coupling strengths.
        """
        self.modes = modes
        self.couplings = couplings
        self._initialize_mode_operators()

    def _initialize_mode_operators(self):
        """
        Initialize transformed system operators for each mode in the quantum system.
        Creates and stores annihilators, number operators, and field operators for each mode.
        """
        self.modes_a = {mode: self._tensor_op(mode, mode.a) for mode in self.modes}
        self.modes_a_dag = {mode: self._tensor_op(mode, mode.a_dag) for mode in self.modes}
        self.modes_num = {mode: self._tensor_op(mode, mode.num) for mode in self.modes}
        self.modes_field = {
            mode: self._tensor_op(mode, mode.field) for mode in self.modes
        }

    def _tensor_op(self, mode: QuantumMode, op: qt.Qobj):
        """
        Tensor an operator with the identity operator on all other modes.

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
        """
        Prepare a tensor product state in the Fock basis for specified modes.

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
        """
        Calculate the expectation value of the population of a mode in a given state.

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
        return f"QuantumSystem({self.modes})"

    @classmethod
    def from_yaml(cls, file_path: str):
        """
        Create a QuantumSystem instance from a YAML file.

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

        # Create QuantumMode instances from modes data
        modes = [QuantumMode(name=k, **v) for k, v in data["modes"].items()]

        # Process couplings
        couplings = {}
        try:
            for k, v in data["couplings"].items():
                mode_names = v["modes"]
                mode_objs = [
                    next(mode for mode in modes if mode.name == name) for name in mode_names
                ]
                couplings[tuple(mode_objs)] = v["g2"]
        except:
            couplings = {}

        return cls(modes, couplings)

    @classmethod
    def from_dict(cls, system_dict: dict):
        """
        Create a QuantumSystem instance from dictionary

        Args:
            system_dict (dict): A dictionary containing modes and couplings data.

        Returns:
            QuantumSystem: An instance of QuantumSystem initialized from the YAML file.
        """
        data = system_dict

        # Create QuantumMode instances from modes data
        modes = [QuantumMode(name=k, **v) for k, v in data["modes"].items()]

        # Process couplings
        couplings = {}
        for k, v in data["couplings"].items():
            mode_names = v["modes"]
            mode_objs = [
                next(mode for mode in modes if mode.name == name) for name in mode_names
            ]
            couplings[tuple(mode_objs)] = v["g2"]

        return cls(modes, couplings)

    def get_mode(self, mode_name: str):
        mode_list = []
        for mode in self.modes:
            if mode.name == mode_name:
                mode_list.append(mode)
        if len(mode_list) > 1:
            raise ValueError("modes in system have duplicate names.")
        elif len(mode_list) == 0:
            raise ValueError(f"No modes named {mode_name} in the system")
        return mode_list[0]


class Hamiltonian:
    def __init__(self, quantum_system: QuantumSystem, use_RWA=True, use_TLS=True):
        """
        Initialize the Hamiltonian for a given quantum system.

        Args:
            quantum_system (QuantumSystem): The quantum system for which the Hamiltonian is constructed.
            use_RWA (bool): Flag to indicate whether to use the Rotating Wave Approximation.
            use_TLS (bool): Flag to indicate whether to use the Two-Level System approximation.
        """
        self.system = quantum_system
        self.use_TLS = use_TLS
        self.use_RWA = use_RWA
        self.H = (
            self._h0RWA() if (use_RWA or self.use_TLS) else self._h0()
        ) + self._Hint()

    def _h0(self):
        r"""
        Generate the linear part of the Hamiltonian.

        The linear frequency term for each mode is represented as:
        :math:`H_0 = 2 \pi f \hat{n}`

        For qubit modes, the Hamiltonian includes the linear frequency term and the anharmonicity term:
        :math:`H_0 = 2 \pi [(f - \alpha) \hat{n} + \frac{\alpha}{12} (\hat{a} + \hat{a}^{\dagger})^4]`

        For SNAIL modes, it includes the linear frequency term and the nonlinearity term:
        :math:`H_0 = 2 \pi [f \hat{n} + \frac{g_3}{6} (\hat{a} + \hat{a}^{\dagger})^3]`
        """
        h0 = 0
        for mode in self.system.modes:
            _num = self.system.modes_num[mode]
            _field = self.system.modes_field[mode]

            # Qubit mode
            if hasattr(mode, "alpha"):
                h0 += 2 * PI * (mode.freq - mode.alpha) * _num
                h0 += 2 * PI * mode.alpha / 12 * _field**4

            # SNAIL mode
            elif hasattr(mode, "g3"):
                h0 += 2 * PI * (mode.freq * _num + mode.g3 / 6 * _field**3)

        return h0

    def _h0RWA(self):
        r"""
        Generate the linear part of the Hamiltonian after applying the Rotating Wave Approximation (RWA).

        For qubit modes:
        :math:`H_{0, \text{RWA}} = 2 \pi [f \hat{n} + \frac{\alpha}{2} \hat{a}^{\dagger} \hat{a}^{\dagger} \hat{a} \hat{a}]`

        For SNAIL modes (preserving only the sss† terms):
        :math:`H_{0, \text{RWA}} = 2 \pi [f \hat{n} + \frac{g_3}{6} (3 \hat{a}^{\dagger} \hat{a} \hat{a} + 3 \hat{a}^{\dagger} \hat{a}^{\dagger} \hat{a})]`
        """
        h0RWA = 0
        for mode in self.system.modes:
            _a = self.system.modes_a[mode]
            _ad = self.system.modes_a_dag[mode]
            _num = self.system.modes_num[mode]

            alpha_term = 0
            g3_term = 0

            # FIXME
            if self.use_TLS and hasattr(mode, "alpha"):
                _sz = _ad * _a - _a * _ad
                h0RWA += 2 * np.pi * _sz / 2
                continue

            # Qubit mode
            if hasattr(mode, "alpha"):
                alpha_term = mode.alpha / 2 * _ad * _ad * _a * _a

            # SNAIL mode
            if hasattr(mode, "g3"):
                g3_term = mode.g3 / 6 * (3 * _ad * _a * _a + 3 * _ad * _ad * _a)

            h0RWA += 2 * np.pi * (mode.freq * _num + alpha_term + g3_term)

        return h0RWA

    def _Hint(self):
        r"""
        Generate the coupling part of the Hamiltonian.

        This includes terms for each coupling in the system:
        :math:`H_{\text{int}} = 2 \pi \sum_{\text{couplings}} g_2 (\hat{a}_m + \hat{a}_m^{\dagger})(\hat{a}_n + \hat{a}_n^{\dagger})`
        where \( g_2 \) is the coupling strength between modes \( m \) and \( n \).
        """
        Hint = 0
        for (mode1, mode2), g2 in self.system.couplings.items():
            _field1 = self.system.modes_field[mode1]
            _field2 = self.system.modes_field[mode2]
            Hint += 2 * np.pi * g2 * _field1 * _field2

        return Hint

class Pulse:
    def __init__(self, omega, amp, phi=0):
        """
        Initialize the Pulses object with common parameters.

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
        return np.exp(-0.5 * ((t - t0 - width / 2) / (width/nsig)) ** 2)

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
        pulse_shape = args.get("shape", Pulse.box)
        shape_params = args.get("shape_params", {})
        envelope = pulse_shape(t, **shape_params)
        return self.amp * np.cos(self.omega * t + self.phi) * envelope

    def plot_pulse(self, pulse_shape, t_list, **shape_params):
        """Plot both the pulse envelope and the modulated pulse."""
        envelope_values = [pulse_shape(t, **shape_params) for t in t_list]
        modulated_values = [
            self.drive(t, {"shape": pulse_shape, "shape_params": shape_params})
            for t in t_list
        ]
        plt.plot(
            t_list, modulated_values, label="Modulated Pulse", linestyle="--", alpha=0.7
        )
        plt.plot(
            t_list, envelope_values, label="Pulse Envelope", linewidth=2, color="red"
        )
        plt.xlabel("Time (dt)")
        plt.ylabel("Amplitude")
        plt.title("Envelope and Modulated Pulse")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    yaml_file = r"E:/projects/20220325_SM/simulation/20231017_evan/system_params_20231205.yaml"
    quantum_system = QuantumSystem.from_yaml(yaml_file)

    # create time-dependent Hamiltonian
    system_hamiltonian = Hamiltonian(quantum_system, use_RWA=False, use_TLS=False)
    # q1_mode = quantum_system.get_mode("q2")  # Q1
    # q2_mode = quantum_system.get_mode("q4")  # Q2
    # snail_mode = quantum_system.get_mode("SNAIL")  # SNAIL
    # snail_field = quantum_system.modes_field[snail_mode]
    #
    # # prepare an initial state
    # psi0 = quantum_system.prepare_tensor_fock_state([(q2_mode, 1)])
    #
    # # Initialize Pulses instance
    # omega_p = 2 * PI * abs(q1_mode.freq - q2_mode.freq)
    # amp_p = 1
    # width_d = 100
    # t_list = np.linspace(0, width_d * 6, 500)
    #
    # pulse = Pulse(omega=omega_p, amp=amp_p)
    #
    # # Additional pulse args used in mesolve
    # args = {"shape": Pulse.smoothbox, "shape_params": {"t0": 0, "width": width_d}}

    # H = [system_hamiltonian.H, [snail_field, pulse.drive]]
    #
    # solve_result = qt.mesolve(H, psi0, t_list, args=args, options=opts, progress_bar=p_bar)
    #
    # # solve_result.states[0]
    # final_state = solve_result.states[-1]
    # print(final_state)
    #
    # # calculate expectation values
    # pi0 = quantum_system.mode_population_expectation(psi0, q2_mode, 0)
    # pf0 = quantum_system.mode_population_expectation(final_state, q2_mode, 0)
    # pi1 = quantum_system.mode_population_expectation(psi0, q2_mode, 1)
    # pf1 = quantum_system.mode_population_expectation(final_state, q2_mode, 1)
    #
    # print(f"pi0 = {pi0:.3f}, pf0 = {pf0:.3f}")
    # print(f"pi1 = {pi1:.3f}, pf1 = {pf1:.3f}")