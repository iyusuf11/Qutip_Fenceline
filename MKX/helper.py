import matplotlib.pyplot as plt
import numpy as np

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


def plot_matrix(Qobj, **kwargs):
    plt.figure()
    plt.pcolormesh(np.real(Qobj.full()), **kwargs)
    plt.colorbar()