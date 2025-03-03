from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from qutip.parallel import parallel_map
from tqdm import tqdm
from hamiltonian import QuantumSystem, Hamiltonian, Pulse

PI = np.pi
rad_to_GHz = 1 / (2 * np.pi)
opts = qt.Options(nsteps=1e6, atol=1e-8, rtol=1e-6)
p_bar = qt.ui.TextProgressBar()


def convert_array(param):
    if type(param) not in [list, np.ndarray]:
        param = np.array([param])
    elif type(param) is list:
        param = np.array(param)
    return param


def get_params_pairs(xparam, yparam):
    xparam = convert_array(xparam)
    yparam = convert_array(yparam)
    params_pairs = [(x, y) for y in yparam for x in xparam]
    return params_pairs


# class Simulation:
#     def __init__(self, quantum_system, hamiltonian, pulse_list):
#         self.quantum_system = quantum_system
#         self.hamiltonian = hamiltonian
#         self.pulse_list = pulse_list
#
#     def simulation_task(self, psi0, pulse, pulse_args):
#         snail_mode = self.hamiltonian.system.get_mode("SNAIL")
#         snail_field = self.hamiltonian.system.modes_field[snail_mode]
#         H_drive = [self.hamiltonian.H, [snail_field, pulse.drive]]
#         solve_result = qt.mesolve(H_drive, psi0, self.t_list, args=pulse_args, options=opts)
#         final_state = solve_result.states[-1]
#         return final_state


def simulation_task(hamiltonian, pulse, psi0, t_list, pulse_args):
    snail_mode = hamiltonian.system.get_mode("SNAIL")
    snail_field = hamiltonian.system.modes_field[snail_mode]
    H_drive = [hamiltonian.H, [snail_field, pulse.drive]]
    solve_result = qt.mesolve(H_drive, psi0, t_list, args=pulse_args, options=opts)
    final_state = solve_result.states[-1]
    return final_state


def simulation_task_amp(amp_tuple, hamiltonian, psi0, t_list, pulse_args):
    amp = amp_tuple
    pulse = Pulse(omega=pulse_args['pulse_params']['freq'], amp=amp)
    final_state = simulation_task(hamiltonian, pulse, psi0, t_list, pulse_args)
    return final_state


def simulation_task_width(width_tuple, hamiltonian, psi0, t_list, pulse_args):
    width_d = width_tuple
    pulse_args['shape_params']['width'] = width_d
    pulse = Pulse(omega=pulse_args['pulse_params']['freq'], amp=pulse_args['pulse_params']['amp'])
    final_state = simulation_task(hamiltonian, pulse, psi0, t_list, pulse_args)
    return final_state


def simulation_task_freq(freq_tuple, hamiltonian, psi0, t_list, pulse_args):
    freq = freq_tuple
    pulse = Pulse(omega=freq, amp=pulse_args['pulse_params']['amp'])
    final_state = simulation_task(hamiltonian, pulse, psi0, t_list, pulse_args)
    return final_state


def simulation_task_freq_width(freq_width_tuple, hamiltonian, psi0, t_list, pulse_args):
    freq, width_d = freq_width_tuple
    pulse_args['shape_params']['width'] = width_d
    pulse = Pulse(omega=freq, amp=pulse_args['pulse_params']['amp'])
    final_state = simulation_task(hamiltonian, pulse, psi0, t_list, pulse_args)
    return final_state


def simulation_task_freq_amp(freq_amp_tuple, hamiltonian, psi0, t_list, pulse_args):
    freq, amp = freq_amp_tuple
    pulse = Pulse(omega=freq, amp=amp)
    final_state = simulation_task(hamiltonian, pulse, psi0, t_list, pulse_args)
    return final_state
