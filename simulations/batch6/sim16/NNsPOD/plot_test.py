import numpy as np
import os
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('WebAgg')

script_dir = os.path.dirname(__file__)
plots_dir = os.path.join(script_dir, 'Plots/')
os.makedirs(plots_dir, exist_ok=True)

interp_dir = os.path.join(script_dir, 'Plots/Interpolation/')
os.makedirs(interp_dir, exist_ok=True)
shift_dir = os.path.join(script_dir, 'Plots/Shift/')
os.makedirs(shift_dir, exist_ok=True)

def clean(array, flag=False, treshold=0.01):

    if flag:
        treshold = 0.1

    for j in range(array.size):

        if array[j] <= treshold:
            array[j] = np.nan

    return array


def interp_plot(idx, input_ref, f_ref, interpolated_f_ref, loss):
    
    x_ref = input_ref[:200,0]
    y_ref = np.zeros(100)
    for k in range(100):

    	y_ref[k] = input_ref[k*200+1,1]

    x, y = np.meshgrid(x_ref, y_ref.T)
    z = f_ref.reshape(y_ref.size, x_ref.size)
    interp_z = interpolated_f_ref.reshape(z.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.contour(x, y, z, colors='green')
    ax.contour(x, y, interp_z, colors='black')

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.view_init(elev=90, azim=270)

    fig_name = 'InterpNet_output{:d}.png'.format(idx)
    plt.suptitle('Epoch [{:d}]: Interpolation loss = {:f}\n\nGreen: Reference snapshot\nBlack: InterpNet output'.format(50*idx, loss), fontsize=8)
    plt.savefig(interp_dir + fig_name)
    

def shift_plot(idx, x_ref, y_ref, f_ref, shifted_x, shifted_y, f_test, loss):

    x_tmp = x_ref[:200]
    shifted_x_tmp = shifted_x[:200]
    y_tmp = np.zeros(100)
    shifted_y_tmp = np.zeros(100)
    for k in range(100):

        y_tmp[k] = y_ref[k*200+1]
        shifted_y_tmp[k] = shifted_y[k*200+1]
        

    x, y = np.meshgrid(x_tmp, y_tmp)
    z = f_ref.reshape(y_tmp.size, x_tmp.size)
    X, Y = np.meshgrid(shifted_x_tmp, shifted_y_tmp)
    Z = f_test.reshape(z.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.contour(x, y, z, colors='green')
    ax.contour(x, y, Z, colors='blue')
    ax.contour(X, Y, Z, colors='red')

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.view_init(elev=90, azim=270)

    fig_name = 'ShiftNet_output{:d}.png'.format(idx)
    plt.suptitle('Epoch [{:d}]: Shift loss = {:f}\n\nGreen: Reference snapshot\nBlue: Test snapshot\nRed: Shifted-test snapshot'.format(idx, loss), fontsize=8)
    plt.savefig(shift_dir + fig_name)
