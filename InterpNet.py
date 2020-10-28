import numpy as np
import torch
import torch.nn as nn
import sys
import os

class InterpNet():

    def __init__(self, input, reference):

        self.func = nn.Sigmoid
        self.lr = 0.001
        self.n_layers = 2
        self.inner_size = 40
        self.epoch = 40000

        self.input = input
        self.reference = reference

        inner_layers = []
        for _ in range(self.n_layers):
            inner_layers.append(nn.Linear(self.inner_size, self.inner_size))
            inner_layers.append(self.func())

        self.model = nn.Sequential(
            nn.Linear(2, self.inner_size), self.func(),
            *inner_layers,
            nn.Linear(self.inner_size, 1),)

        self.criterion = torch.nn.MSELoss(reduction = 'mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

        self.train()


    def forward(self):

        return self.model(self.input)


    def train(self):

        for epoch in range(self.epoch):

            output = self.forward()
            loss = self.criterion(self.reference, output)

            if epoch == 0:

                fio.write("\nInterpNet loss starts at {:f}.".format(loss.item()))

            if epoch % 100 == 0:

                print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  loss.item()))

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        self.save()
        fio.write("\nInterpNet loss after {:d} epochs: {:f}\n.".format(epoch, loss.item()))


    def save(self):

        torch.save(self.model, 'InterpNet.pt')


ref = 75
test = 0

fio = open("Performances(ref.snap={:d}).txt".format(ref), "w+")
fio.write("The reference is the {:d}-th snapshot.\n".format(ref))

timesteps = np.genfromtxt("timesteps.txt")

Ns = timesteps.size

timesteps = timesteps

f = []
x = []
y = []

for j in range(Ns):

    snap_name = "./{:d}.npy".format(j)
    X = np.load(snap_name)

    f.append(X[:,0])
    x.append(X[:,1])
    y.append(X[:,2])

Nh = X[:,0].size

f = torch.tensor(np.array(f), requires_grad=True).T # f.shape = torch.Size([Nh, Ns])
x = torch.tensor(np.array(x), requires_grad=True).T # x.shape = torch.Size([Nh, Ns])
y = torch.tensor(np.array(y), requires_grad=True).T # y.shape = torch.Size([Nh, Ns])

f_ref = f[:,ref].clone().float().reshape(-1, 1) # f_ref.shape = torch.Size([Nh, 1])
x_ref = x[:,ref].clone().float().reshape(-1, 1) # x_ref.shape = torch.Size([Nh, 1])
y_ref = y[:,ref].clone().float().reshape(-1, 1) # y_ref.shape = torch.Size([Nh, 1])

input_ref = torch.cat((x_ref, y_ref), 1) # input_ref.shape = torch.Size([Nh, 2])
# np.savetxt("input_ref", input_ref.detach().numpy(), newline="\n")

trained_interpolation = InterpNet(input_ref, f_ref)