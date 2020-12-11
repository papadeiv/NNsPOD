from plot_test import interp_plot
import numpy as np
import torch
import torch.nn as nn
import sys
import os

class InterpNet():

    def __init__(self):

        self.func = nn.Sigmoid
        self.lr = 0.0001
        self.n_layers = 4
        self.inner_size = 40
        self.epoch = 45000

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


    def build_input(self, x, y):

        return torch.cat((x, y), 1)


    def forward(self, coordinates):

        return self.model(coordinates)


    def train(self, coordinates, reference):

        plot_counter = 0

        for epoch in range(self.epoch):

            output = self.forward(coordinates)
            loss = self.criterion(reference, output)

            if epoch == 0:

                with open('./Results/Training performance.txt', 'a') as f:

                    f.write("\nInterpNet loss starts at {:f}.".format(loss.item()))

            if epoch % 50 == 0:

                self.save()

                print('[epoch {:4d}] {:18.8f}'.format(epoch, loss.item()))

                interp_plot(plot_counter, coordinates.clone().detach().numpy()
                                        , reference.clone().detach().numpy()
                                        , output.clone().detach().numpy(), loss.item())
                plot_counter += 1

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        self.save()
        f = open('./Results/Training performance.txt', 'a')
        f.write("\nInterpNet loss after {:d} epochs: {:f}.\n\n".format(epoch, loss.item()))
        f.close()


    def save(self):

        torch.save(self.model, 'InterpNet.pt')
