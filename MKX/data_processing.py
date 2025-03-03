import h5py
import os
from pathlib import Path
import yaml
import ast
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
from hamiltonian import QuantumSystem, Hamiltonian


def save_data(file_path, overwrite=0, **kwargs):
    idx = max([file_path.rfind("\\"), file_path.rfind("/")])
    file_dir = file_path[:idx+1]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not overwrite:
        if os.path.exists(file_path):
            raise OSError("the file already exists!")
    hf = h5py.File(file_path, 'w')
    for key, value in kwargs.items():
        hf.create_dataset(key, data=value)
    hf.close()


def load_data(file_path):
    hf = h5py.File(file_path, 'r')
    data = {}
    system_info = {}
    for k, v in hf.items():
        if k != "system_info":
            data[k] = v[()]
        else:
            system_info = ast.literal_eval(v[()].decode())
    return data, system_info


def load_yaml(file_path):
    yaml_path = Path(file_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found at: {yaml_path}")

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def plot_data_raw2D(data, system_info, dx, dy, xlabel=None, ylabel=None):
    quantum_system = QuantumSystem.from_dict(system_info)
    system_hamiltonian = Hamiltonian(quantum_system, use_RWA=True, use_TLS=False)
    h_dims = system_hamiltonian.H.dims
    state_dims = [h_dims[0], [1]*len(h_dims[0])]

    fig, axes = plt.subplots(1, len(quantum_system.modes), figsize=(len(quantum_system.modes) * 5, 4))
    for i, param in enumerate(tqdm(data['pulse_params'])):
        state = qt.Qobj(data['final_states'][i])
        state.dims = state_dims
        populations = [quantum_system.mode_population_expectation(state, mode, 0) for mode in
                       quantum_system.modes]
        x_ = [param[0] - dx/2, param[0] + dx/2]
        y_ = [param[1] - dy/2, param[1] + dy/2]
        # print(x_, y_)
        for j, ax in enumerate(axes):
            ax.pcolor(x_, y_, [[populations[j]]], vmin=0, vmax=1, shading='auto')
    sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    for j, ax in enumerate(axes):
        ax.set_title(f"{quantum_system.modes[j].name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(sm, ax=axes[j], orientation="vertical", label="|g> Population")
    plt.tight_layout()
    return fig, axes


def plot_data_1D(xdata, qdata, quantum_system, vmin=0, vmax=1, xlabel=None):
    fig, axes = plt.subplots(len(quantum_system.modes), 1, figsize=(5, 6))
    n_xticks = 5
    for k, mode in enumerate(quantum_system.modes):
        print()
        axes[k].plot(xdata, qdata[k].flatten())
        axes[k].set_ylim([vmin, vmax])

        axes[k].set_title(f"Mode: {mode.name}")
        axes[k].set_xlabel(xlabel)
        axes[k].set_ylabel("population")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_data_2D(xdata, ydata, qdata, quantum_system, vmin=0, vmax=1, xlabel=None, ylabel=None):
    fig, axes = plt.subplots(1, len(quantum_system.modes), figsize=(len(quantum_system.modes) * 5, 4))

    # Define the number of ticks for the detuning x-axis
    n_xticks = 5  # Adjust the number of ticks as needed

    # Plot data for each mode
    for k, mode in enumerate(quantum_system.modes):
        im = axes[k].imshow(
            qdata[k],
            extent=[xdata[0], xdata[-1], ydata[0], ydata[-1]],
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,  # Set color bar scale from 0.0 to 1.0
        )
        axes[k].set_title(f"Mode: {mode.name}")
        axes[k].set_xlabel(xlabel)
        axes[k].set_ylabel(ylabel)

        # Customize x-axis ticks
        xticks = np.linspace(xdata[0], xdata[-1], n_xticks)
        axes[k].set_xticks(xticks)
        axes[k].set_xticklabels([f"{x:.5f}" for x in xticks])  # Format the tick labels

        # Add color bar
        cbar = fig.colorbar(
            im, ax=axes[k], orientation="vertical", label="|g> Population"
        )
        cbar.set_ticks(np.linspace(vmin, vmax, 3))  # Customize color bar ticks

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_data(xdata, ydata, qdata, quantum_system, vmin=0, vmax=1, xlabel=None, ylabel=None):
    if len(xdata) > 1 and len(ydata) > 1:
        fig, axes = plot_data_2D(xdata, ydata, qdata, quantum_system, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel)
    elif len(xdata) == 1:
        fig, axes = plot_data_1D(ydata, qdata, quantum_system, vmin=vmin, vmax=vmax, xlabel=ylabel)
    elif len(ydata) == 1:
        fig, axes = plot_data_1D(xdata, qdata, quantum_system, vmin=vmin, vmax=vmax, xlabel=xlabel)
    return fig, axes

