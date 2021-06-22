from plot_test import interp_plot
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import csv
import math

class InterpNet():

    def __init__(self):

        self.func = nn.Sigmoid
        self.lr = 1e-3
        self.n_layers = 2
        self.inner_size = 40 
        self.epoch = int(1e6)
        self.treshold = 1e-7
        self.plot_counter = 0

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

        with open('./Results/InterpNetLoss.csv', 'w+', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['Epoch', 'Loss'])

        for epoch in range(self.epoch):

            output = self.forward(coordinates)
            loss = self.criterion(reference, output)
            magnitude = pow(10, math.floor(math.log10(epoch+1)))

            if epoch == 0:

                print('[Epoch {:4d}] {:18.8f}'.format(epoch, loss.item()))

                interp_plot(self.plot_counter, epoch, coordinates.clone().detach().numpy()
                                        , reference.clone().detach().numpy()
                                        , output.clone().detach().numpy(), loss.item())
                self.plot_counter += 1

            if epoch % 100 == 0:

            	self.save()

            if epoch % magnitude == 0 and epoch >= 10:

                print('[Epoch {:4d}] {:18.8f}'.format(epoch, loss.item()))

                interp_plot(self.plot_counter, epoch, coordinates.clone().detach().numpy()
                                        , reference.clone().detach().numpy()
                                        , output.clone().detach().numpy(), loss.item())
                self.plot_counter += 1

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            with open('./Results/InterpNetLoss.csv', 'a', newline='') as csvf:
                writer = csv.writer(csvf)
                writer.writerow([epoch, loss.item()])

            if loss.item() < self.treshold:

            	print("\n**** InterpNet's accuracy treshold reached ****\n")
            	interp_plot(self.plot_counter, epoch, coordinates.clone().detach().numpy()
                                        , reference.clone().detach().numpy()
                                        , output.clone().detach().numpy(), loss.item())

            	break

        print('Training completed: saving the full model (weights, biases and architecture) for generalisation\n\n')
        self.save()


    def save(self):

        script_dir = os.path.dirname(__file__)
        res_dir = os.path.join(script_dir, 'TrainedModels/')
        os.makedirs(res_dir, exist_ok=True)

        torch.save(self.model, './TrainedModels/InterpNet.pt')