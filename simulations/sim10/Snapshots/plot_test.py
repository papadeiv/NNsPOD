import numpy as np
import os
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('WebAgg')

script_dir = os.path.dirname(__file__)
res_dir = os.path.join(script_dir, 'Plots/')
os.makedirs(res_dir)

interp_dir = os.path.join(script_dir, 'Plots/Interpolation/')
os.makedirs(interp_dir)
shift_dir = os.path.join(script_dir, 'Plots/Shift/')
os.makedirs(shift_dir)

def clean(array, flag=False, treshold=0.01):

    if flag:
        treshold = 0.1

    for j in range(array.size):

        if array[j] <= treshold:
            array[j] = np.nan

    return array


def interp_plot(idx, input_ref, f_ref, interpolated_f_ref, loss):
    
    x_ref, y_ref = input_ref[:50,0], input_ref[:50,1]
    y_ref = x_ref

    x, y = np.meshgrid(x_ref, y_ref)
    z = clean(f_ref).reshape(x_ref.size, y_ref.size)
    interp_z = interpolated_f_ref.reshape(z.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.contour(x, y, z, colors='green')
    ax.contour(x, y, interp_z, colors='black')

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.view_init(elev=90, azim=270)

    fig_name = 'InterpNet_output{:d}.png'.format(idx)
    plt.suptitle('Epoch [{:d}]: Interpolation loss = {:f}\n\nGreen: Reference snapshot\nBlack: InterpNet output'.format(100*idx, loss), fontsize=8)
    plt.savefig(interp_dir + fig_name)
    

def shift_plot(idx, x_ref, y_ref, f_ref, shifted_x, shifted_y, f_test, loss):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(x_ref, y_ref, clean(f_ref, flag=True), color="g", label="Reference")
    ax.plot_trisurf(x_ref, y_ref, clean(f_test, flag=True), color="b", label="Test")
    ax.plot_trisurf(shifted_x.flatten(), shifted_y.flatten(), clean(f_test, flag=True), color="r", label="Shifted-test")

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.view_init(elev=70, azim=270)

    fig_name = 'InterpNet_output{:d}.png'.format(idx)
    plt.suptitle('Epoch [{:d}]: Shift loss = {:f}\n\nGreen: Reference snapshot\nBlue: Test snapshot\nRed: Shifted-test snapshot'.format(100*idx, loss), fontsize=8)
    plt.savefig(shift_dir + fig_name)
