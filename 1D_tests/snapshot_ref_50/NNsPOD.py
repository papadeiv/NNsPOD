import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import sys
import os

################# ASSEMBLY OF THE DATASET #################

ref = 50

fio = open("Performances({:d}).txt".format(ref), "w+")
fio.write("The reference is the {:d}-th snapshot".format(ref))
fio.write("\n")

dt = 0.0784
timesteps = np.arange(1, 205) * dt

x_db = []
y_db = []

for i in range(1, 205):
    fname = 'data_wave_2/wave.{:d}.csv'.format(i)
    y, x = np.genfromtxt(fname, usecols=(0, 1), delimiter=',', skip_header=1).T
    # fname = 'data_wave_1/ffffff_{:d}.csv'.format(i)
    # y, x = np.genfromtxt(fname, usecols=(1, 2), delimiter=',', skip_header=1).T
    y_db.append(y)
    x_db.append(x)
y_db = torch.tensor(y_db, requires_grad=True).T # x_db.shape = torch.Size([200, 204])
x_db = torch.tensor(x_db, requires_grad=True).T # y_db.shape = torch.Size([200, 204])

################# INTERPOLATION NEURAL NET #################

class InterpNet():

    def __init__(self, input, output):

        self.func = nn.Sigmoid
        self.lr = 0.1
        self.n_layers = 2
        self.inner_size = 20
        self.epoch = 10

        inner_layers = []
        for _ in range(self.n_layers):
            inner_layers.append(nn.Linear(self.inner_size, self.inner_size))
            inner_layers.append(self.func())

        self.model = nn.Sequential(
            nn.Linear(1, self.inner_size), self.func(),
            *inner_layers,
            nn.Linear(self.inner_size, 1),)

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epoch):
            y_pred = self.model(input)
            lss = self.criterion(output, y_pred)
            if epoch % 10 == 0:
                print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  lss.item()))

            self.model.zero_grad()
            lss.backward(retain_graph=True)
            self.optimizer.step()

        fio.write("Intepolation loss at {:d} epochs: {:f}".format(epoch, lss.item()))

    def forward(self, input):
        return self.model(input)


x_ref = x_db.T[ref].clone().float().reshape(-1, 1) # x_ref.shape = torch.Size([200, 1])
y_ref = y_db.T[ref].clone().float().reshape(-1, 1) # y_ref.shape = torch.Size([200, 1])

trained_interpolation = InterpNet(x_ref, y_ref)

################# SHIFT-DETECTING NEURAL NET #################

inner_size = 10

inner_layers = []
for _ in range(1):
    inner_layers.append(nn.Linear(inner_size, inner_size))
    inner_layers.append(nn.Sigmoid())

ShiftNet = nn.Sequential(
    nn.Linear(2, inner_size), nn.Sigmoid(),
    *inner_layers,
    nn.Linear(inner_size, 1),)

dyn_lr = 0.01

optimizer = torch.optim.Adam(ShiftNet.parameters(), lr=dyn_lr)

def test_plot(idx):

    idx_test = 0
    t = timesteps[idx_test]
    x_test = x_db.T[idx_test]
    y_test = y_db.T[idx_test]
    input_ = torch.cat((
        x_test.reshape(-1, 1).float(),
        torch.ones((x.shape[0], 1), dtype=torch.float, requires_grad=True)*t),
        axis=-1)

    shift = ShiftNet(input_)
    shifted_x = x_test.reshape(shift.shape) - shift

    plt.cla()
    plt.plot(x_test.detach(), y_test.detach(), '-b', label='Test snapshot')
    plt.plot(shifted_x.detach(), y_test.detach(), '--r', label='Shifted snapshot')
    plt.plot(x_ref.detach(), y_ref.detach(), '--g', label='Refernce snapshot')
    plt.suptitle('The reference is the 150-th snapshot; the test is the IC snapshot', fontsize=16)
    plt.legend()

    plt.savefig('training{:d}.png'.format(idx))

################# TRAINING PHASE #################

previous_loss = sys.float_info.max
lr_update = 0
idx_plot = 0

for epoch in range(300):

    def compute_loss():
        optimizer.zero_grad()

        loss = 0.0

        for x, y, t in zip(x_db.T, y_db.T, timesteps):

            # x.shape = y.shape = torch.Size([200])

            input_ = torch.cat((
                x.reshape(-1, 1).float(),
                torch.ones((x.shape[0], 1), dtype=torch.float, requires_grad=True)*t),
                axis=-1)

            # input_.shape = torch.Size([200, 2])

            shift = ShiftNet(input_)

            # shift.shape = torch.Size([200, 1])

            shifted_x = x.reshape(-1, 1) - shift

            # shifted_x.shape = torch.Size([200, 1])

            shifted_y = trained_interpolation.forward(shifted_x.float()).flatten()

            loss += torch.sum(torch.abs((shifted_y.flatten() - y)))

        loss.backward()
        return loss


    loss = optimizer.step(compute_loss)

    if (loss - previous_loss>0.1*loss and dyn_lr>=0.00000001):
        lr_update += 1
        fio.write("\n")
        fio.write("Epoch [{:d}]    Learning rate updated from {:f} ".format(epoch, dyn_lr))
        dyn_lr = 0.75*dyn_lr
        fio.write("to {:f}".format(dyn_lr))
        fio.write("\n")

    previous_loss = loss

    if epoch % 100 == 0:
        print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  loss.item()))

    idx_plot +=1
    test_plot(idx_plot)

fio.write("Shift loss at {:d} epochs: {:f}".format(epoch, loss.item()))
fio.write("\n")
fio.write("The learning rate has been updated {:d} times".format(lr_update))
fio.close()
