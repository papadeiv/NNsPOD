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
        treshold = 0.005

    for j in range(array.size):

        if array[j] <= treshold:
            array[j] = np.nan

    return array


def interp_plot(idx, input_ref, f_ref, interpolated_f_ref, loss):
    
    x_ref = input_ref[:50,0]
    y_ref = np.zeros(50)
    for k in range(50):

    	y_ref[k] = input_ref[k*50+1,1]

    x, y = np.meshgrid(x_ref, y_ref.T)
    z = clean(f_ref,flag=True).reshape(y_ref.size, x_ref.size)
    interp_z = interpolated_f_ref.reshape(z.shape)

    fig = plt.figure(figsize=(12,10),dpi=350.0)

    ax = fig.gca(projection='3d')

    ax.contour(x, y, z, colors='green')
    ax.contour(x, y, interp_z, colors='black')

    #ax.set_xlabel('x', fontsize=10)
    ax.axes.xaxis.set_ticklabels([])
    #ax.set_ylabel('y', fontsize=10)
    ax.axes.yaxis.set_ticklabels([])
    ax.set_zticks([])
    ax.view_init(elev=90, azim=270)

    fig_name = 'InterpNet_output{:d}.png'.format(idx)
    plt.suptitle('Interpolation loss (epoch = {:d}) = {:f}\n\nGreen: Reference configuration (80-th snapshot)\nBlack: InterpNet output'.format(10*idx, loss), fontsize=18, x=0.5, y=0.9)
    plt.savefig(interp_dir + fig_name)
    

def shift_plot(idx, x_ref, y_ref, f_ref, shifted_x, shifted_y, f_test, loss):

    x_tmp = x_ref[:50]
    shifted_x_tmp = shifted_x[:50]
    y_tmp = np.zeros(50)
    shifted_y_tmp = np.zeros(50)
    for k in range(50):

        y_tmp[k] = y_ref[k*50+1]
        shifted_y_tmp[k] = shifted_y[k*50+1]
        

    x, y = np.meshgrid(x_tmp, y_tmp)
    z = clean(f_ref,flag=True).reshape(y_tmp.size, x_tmp.size)
    X, Y = np.meshgrid(shifted_x_tmp, shifted_y_tmp)
    Z = f_test.reshape(z.shape)

    fig = plt.figure(figsize=(12,10),dpi=350.0)
    ax = fig.add_subplot(111, projection='3d')

    ax.contour(x, y, z, colors='green')
    ax.contour(x, y, Z, colors='blue')
    ax.contour(X, Y, Z, colors='red')

    #ax.set_xlabel('x', fontsize=10)
    ax.axes.xaxis.set_ticklabels([])
    #ax.set_ylabel('y', fontsize=10)
    ax.axes.yaxis.set_ticklabels([])
    ax.set_zticks([])
    ax.view_init(elev=90, azim=270)

    fig_name = 'ShiftNet_output{:d}.png'.format(idx)
    plt.suptitle('Shift loss (epoch = {:d}) = {:f}\n\nGreen: Reference configuration (80-th snapshot)\nRed: ShiftNet output (test snapshot)'.format(20*idx, loss), fontsize=18, x=0.5, y=0.9)
    plt.savefig(shift_dir + fig_name)
