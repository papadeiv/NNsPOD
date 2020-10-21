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

timesteps = np.genfromtxt("timesteps.txt")

Ns = timesteps.size

f = []
x = []
y = []

for j in range(Ns):

    snap_name = "./{:d}.npy".format(j)
    X = np.load(snap_name)

    f.append(X[:,0])
    x.append(X[:,1])
    y.append(X[:,2])

Nh = f[ref].size

fio.write("Number of snapshots: {:d}\nFull-order (snapshot) dimension: {:d}\n".format(Ns, Nh))
fio.write("\n")
print("Number of snapshots: {:d}\nFull-order dimension: {:d}\n".format(Ns, Nh))

f = torch.tensor(f, requires_grad=True)
x = torch.tensor(x, requires_grad=True)
y = torch.tensor(y, requires_grad=True)

################# INTERPOLATION NEURAL NET #################

class InterpNet():

    def __init__(self, input, output):

        self.func = nn.Sigmoid
        self.lr = 0.1
        self.n_layers = 2
        self.inner_size = 20
        self.epoch = 100

        inner_layers = []
        for _ in range(self.n_layers):
            inner_layers.append(nn.Linear(self.inner_size, self.inner_size))
            inner_layers.append(self.func())

        self.model = nn.Sequential(
            nn.Linear(2, self.inner_size), self.func(),
            *inner_layers,
            nn.Linear(self.inner_size, 1),)

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epoch):
            y_pred = self.model(input)
            lss = self.criterion(output, y_pred)

            if epoch == 0:
                fio.write("InterpNet loss starts at {:f}".format(lss.item()))

            if epoch % 10 == 0:
                print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  lss.item()))

            self.model.zero_grad()
            lss.backward(retain_graph=True)
            self.optimizer.step()

        fio.write("InterpNet loss at {:d} epochs: {:f}".format(epoch, lss.item()))

    def forward(self, input):
        return self.model(input)

f_ref = f.T[ref].clone().float().reshape(-1, 1)
x_ref = x.T[ref].clone().float().reshape(-1, 1)
y_ref = y.T[ref].clone().float().reshape(-1, 1)
input_ref = torch.cat((x_ref, y_ref), 1)
trained_interpolation = InterpNet(input_ref, f_ref)

################# SHIFT-DETECTING NEURAL NET #################

inner_size = 40

inner_layers = []
for _ in range(3):
    inner_layers.append(nn.Linear(inner_size, inner_size))
    inner_layers.append(nn.Sigmoid())

ShiftNet = nn.Sequential(
    nn.Linear(3, inner_size), nn.Sigmoid(),
    *inner_layers,
    nn.Linear(inner_size, 2),)

dyn_lr = 1.0

optimizer = torch.optim.Adam(ShiftNet.parameters(), lr=dyn_lr)

def test_plot(idx):

    idx_test = 0
    t = timesteps[idx_test]
    x_test = x[idx_test]
    y_test = y[idx_test]

    input_ = torch.cat((x_test, y_test, t), 1)

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

for epoch in range(1000):

    def compute_loss():
        optimizer.zero_grad()

        loss = 0.0

        for j, k, t, l in zip(x, y, timesteps, f):

            # j.shape = k.shape = f.shape = torch.Size([2500])

            input_ = torch.cat((j.reshape(-1, 1).float(), k.reshape(-1, 1).float(), torch.ones((j.shape[0],1), dtype=torch.float, requires_grad=True)*t), -1)

            # input_.shape = torch.Size([2500, 3])

            shift = ShiftNet(input_)

            # shift.shape = torch.Size([2500, 2])

            shifted_x = j - shift[:, 0]
            shifted_y = k - shift[:, 1]

            # shifted_x.shape = shifted_y.shape = torch.Size([2500])

            shifted_input = torch.cat((shifted_x.reshape(-1, 1), shifted_y.reshape(-1, 1)), 1)

            # shifted_input.shape = torch.Size([2500, 2])

            shifted_f = trained_interpolation.forward(shifted_input.float())

            # shifted_f.shape = torch.Size([2500, 1])

            shifted_f = shifted_f.flatten()

            # shifted_f.shape = torch.Size([2500])

            loss += torch.sum(torch.abs((shifted_f - l)))

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
        fio.write("\n")
        fio.write("Epoch [{:d}]    ShiftNet loss = {:f} ".format(epoch, dyn_lr))

    idx_plot +=1
    # test_plot(idx_plot)

fio.write("Shift loss at {:d} epochs: {:f}".format(epoch, loss.item()))
fio.write("\n")
fio.write("The learning rate has been updated {:d} times".format(lr_update))
fio.close()