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

timesteps = timesteps.reshape(-1, 1)

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

f = torch.tensor(f, requires_grad=True).T
# f.shape = torch.Size([Nh, Ns])
x = torch.tensor(x, requires_grad=True).T
# x.shape = torch.Size([Nh, Ns])
y = torch.tensor(y, requires_grad=True).T
# y.shape = torch.Size([Nh, Ns])

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
                fio.write("\n")

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
# f_ref.shape = torch.Size([Nh, 1])

x_ref = x.T[ref].clone().float().reshape(-1, 1)
# np.savetxt("x_ref", x_ref.detach().numpy(), newline="\n")
# x_ref.shape = torch.Size([Nh, 1])

y_ref = y.T[ref].clone().float().reshape(-1, 1)
# np.savetxt("y_ref", y_ref.detach().numpy(), newline="\n")
# y_ref.shape = torch.Size([Nh, 1])

# x_ref = x_ref[Nh-3:]
# y_ref = y_ref[Nh-3:]

input_ref = torch.cat((x_ref, y_ref), 1)
# np.savetxt("input_ref", input_ref.detach().numpy(), newline="\n")
# input_ref.shape = torch.Size([Nh, 2])

trained_interpolation = InterpNet(input_ref, f_ref)

################# SHIFT-DETECTING NEURAL NET #################

inner_size = 10

inner_layers = []
for _ in range(1):
    inner_layers.append(nn.Linear(inner_size, inner_size))
    inner_layers.append(nn.Sigmoid())

ShiftNet = nn.Sequential(
    nn.Linear(3, inner_size), nn.Sigmoid(),
    *inner_layers,
    nn.Linear(inner_size, 2),)

dyn_lr = 0.1

optimizer = torch.optim.Adam(ShiftNet.parameters(), lr=dyn_lr)

def test_plot(idx):

    idx_test = 0
    t = timesteps[idx_test]
    x_test = x.T[idx_test]
    x_test = x_test[2200:2250]
    y_test = y.T[idx_test]
    y_test = y_test[2200:2250]
    f_test = f.T[idx_test]
    f_test = f_test[2200:2250]

    x_new_ref = x_ref[1900:1950]
    y_new_ref = y_ref[1900:1950]
    f_new_ref = f_ref[1900:1950]

    np.savetxt("x_test", x_test.detach().numpy(), newline="\n")
    np.savetxt("x_ref", x_new_ref.detach().numpy(), newline="\n")
    np.savetxt("y_test", y_test.detach().numpy(), newline="\n")
    np.savetxt("y_ref", y_new_ref.detach().numpy(), newline="\n")
    np.savetxt("f_test", f_test.detach().numpy(), newline="\n")
    np.savetxt("f_ref", f_new_ref.detach().numpy(), newline="\n")

    input_ = torch.cat((
        x_test.reshape(-1, 1).float(), y_test.reshape(-1, 1).float(), 
        torch.ones((x_test.shape[0], 1), dtype=torch.float, requires_grad=True)*t.item()), 
        axis=-1)

    shift = ShiftNet(input_)

    shifted_x = x_test.reshape(-1, 1) - shift[:, 0].reshape(-1, 1)
    shifted_x = shifted_x.reshape(-1, 1)

    shifted_y = y_test.reshape(-1, 1) - shift[:, 1].reshape(-1, 1)
    shifted_y = shifted_y.reshape(-1, 1)

    plt.cla()
    plt.plot(x_test.detach(), f_test.detach(), '-b', label='Test snapshot')
    plt.plot(shifted_x.detach(), f_test.detach(), '--r', label='Shifted snapshot')
    plt.plot(x_new_ref.detach(), f_new_ref.detach(), '--g', label='Refernce snapshot')
    plt.suptitle('The reference is the 100-th snapshot; the test is the IC snapshot', fontsize=10)
    plt.legend()

    plt.savefig('training{:d}.png'.format(idx))

################# TRAINING PHASE #################

previous_loss = sys.float_info.max
lr_update = 0
idx_plot = 0

for epoch in range(100):

    def compute_loss():
        optimizer.zero_grad()

        loss = 0.0

        idx_t = 0

        for j, k, t, l in zip(x.T, y.T, timesteps, f.T):

            if(idx_t==ref or idx_t==0):
                np.savetxt("f({:d})".format(idx_t), l.detach().numpy(), newline="\n")

            idx_t += 1

            # j.shape = k.shape = f.shape = torch.Size([Nh])

            input_ = torch.cat((
                j.reshape(-1, 1).float(), k.reshape(-1, 1).float(), 
                torch.ones((j.shape[0],1), dtype=torch.float, requires_grad=True)*t.item()), 
                axis=-1)

            # input_.shape = torch.Size([Nh, 3])

            shift = ShiftNet(input_)

            # shift.shape = torch.Size([Nh, 2])

            shifted_x = j.reshape(-1, 1) - shift[:, 0].reshape(-1, 1)
            shifted_y = k.reshape(-1, 1) - shift[:, 1].reshape(-1, 1)

            # shifted_x.shape = shifted_y.shape = torch.Size([Nh, 1])

            shifted_input = torch.cat((shifted_x.reshape(-1, 1), shifted_y.reshape(-1, 1)), 1)

            # shifted_input.shape = torch.Size([Nh, 2])

            shifted_f = trained_interpolation.forward(shifted_input.float())

            # shifted_f.shape = torch.Size([Nh, 1])

            shifted_f = shifted_f.flatten()

            # shifted_f.shape = torch.Size([Nh])

            loss += torch.sum(torch.abs((shifted_f - l)))

        loss = loss/Ns
        loss.backward()
        return loss


    loss = optimizer.step(compute_loss)

    if (loss - previous_loss >= np.finfo(float).eps**4 and dyn_lr>=0.00000001):
        lr_update += 1
        fio.write("\n")
        fio.write("Epoch [{:d}]    Learning rate updated from {:f} ".format(epoch, dyn_lr))
        dyn_lr = 0.75*dyn_lr
        fio.write("to {:f}".format(dyn_lr))
        fio.write("\n")
        print(optimizer.param_groups[0]['lr'])
        for g in optimizer.param_groups:

            g['lr'] = dyn_lr

        print(optimizer.param_groups[0]['lr'])


    previous_loss = loss

    if epoch % 1 == 0:
        print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  loss.item()))
        fio.write("\n")
        fio.write("Epoch [{:d}]    ShiftNet loss = {:f} ".format(epoch, loss))

    test_plot(idx_plot)
    idx_plot +=1

fio.write("\n")
fio.write("The learning rate has been updated {:d} times".format(lr_update))
fio.close()

################# GENERALIZATION AND OUTPUT #################

X = np.zeros((Nh, Ns))
counter = 0

for j, k, t in zip(x, y, timesteps):

    input_ = torch.cat((j.reshape(-1, 1).float(), k.reshape(-1, 1).float(), torch.ones((j.shape[0],1), dtype=torch.float, requires_grad=True)*t), -1)

    shift = ShiftNet(input_)

    shifted_x = j - shift[:, 0]
    shifted_y = k - shift[:, 1]

    shifted_input = torch.cat((shifted_x.reshape(-1, 1), shifted_y.reshape(-1, 1)), 1)

    shifted_f = trained_interpolation.forward(shifted_input.float())

    shifted_f = shifted_f.flatten()

    snapshot = shifted_f.clone()

    snapshot = snapshot.detach().numpy()

    X[counter, :] = snapshot

    counter+=1

np.save("shifted_snapshot_matrix({:d})".format(ref), X)