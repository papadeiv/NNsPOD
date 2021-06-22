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

x_cells = 50
y_cells = 50


def clean(array, flag=False, treshold=0.1):

    if flag:
        treshold = 0.0005

    for j in range(array.size):

        if array[j] <= treshold:
            array[j] = np.nan

    return array


def interp_plot(idx, epoch, input_ref, f_ref, interpolated_f_ref, loss):
    
    x_ref = input_ref[:x_cells,0]
    y_ref = np.zeros(y_cells)
    for k in range(y_cells):

    	y_ref[k] = input_ref[k*x_cells+1,1]

    x, y = np.meshgrid(x_ref, y_ref.T)
    z = clean(f_ref,flag=True).reshape(y_ref.size, x_ref.size)
    interp_z = interpolated_f_ref.reshape(z.shape)

    fig = plt.figure(figsize=(10,8),dpi=300.0)

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
    plt.suptitle('[Epoch {:4d}]   Loss = {:f}'.format(epoch, loss), fontsize=18, x=0.5, y=0.8)
    plt.savefig(interp_dir + fig_name)
    

def shift_plot(idx, epoch, x_ref, y_ref, f_ref, shifted_x, shifted_y, f_test, loss):

    x_tmp = x_ref[:x_cells]
    shifted_x_tmp = shifted_x[:x_cells]
    y_tmp = np.zeros(y_cells)
    shifted_y_tmp = np.zeros(y_cells)
    for k in range(y_cells):

        y_tmp[k] = y_ref[k*x_cells+1]
        shifted_y_tmp[k] = shifted_y[k*x_cells+1]
        

    x, y = np.meshgrid(x_tmp, y_tmp)
    z = clean(f_ref, flag=True).reshape(y_tmp.size, x_tmp.size)
    X, Y = np.meshgrid(shifted_x_tmp, shifted_y_tmp)
    Z = clean(f_test, flag=True).reshape(y_tmp.size, x_tmp.size)

    fig = plt.figure(figsize=(10,8),dpi=300.0)
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
    plt.suptitle('[Epoch {:4d}]   Loss = {:f}'.format(epoch, loss), fontsize=18, x=0.5, y=0.8)
    plt.savefig(shift_dir + fig_name)
