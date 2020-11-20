from interp_net import InterpNet
from plot_test import shift_plot
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import csv

class ShiftNet():

    def __init__(self, ref, test):

        self.func = nn.Sigmoid
        self.lr = 0.0001
        self.n_layers = 3
        self.inner_size = 20
        self.epoch = 2#0000

        self.ref = ref
        self.test = test

        inner_layers = []
        for _ in range(self.n_layers):
            inner_layers.append(nn.Linear(self.inner_size, self.inner_size))
            inner_layers.append(nn.Sigmoid())

        self.model = nn.Sequential(
            nn.Linear(3, self.inner_size), nn.Sigmoid(),
            *inner_layers,
            nn.Linear(self.inner_size, 2),)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def build_input(self, x, y, t):

        snap_t = torch.tensor(t*np.ones((x.shape[0], 1), dtype=float), requires_grad=True) # snap_t.shape = torch.Size([Nh, 1])
        snap_x = x.reshape(-1, 1)
        snap_y = y.reshape(-1, 1)

        coordinates = torch.cat((snap_x.float(), snap_y.float(), snap_t.float()), 1)

        return coordinates


    def forward(self, coordinates):

        return self.model(coordinates)


    def train(self, X, Y, Ts, F):

        with open('./Results/Loss.csv', 'w+', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['Epoch', 'Loss'])

        Ns = Ts.size
        plot_counter = 0

        for epoch in range(self.epoch):

            loss = 0.0
            self.optimizer.zero_grad()

            snap_counter = 0

            for x, y, t, f in zip(X.T, Y.T, Ts, F.T):

                coordinates = self.build_input(x,y,t)

                shift = self.forward(coordinates)

                shift_x = (shift[:, 0]).reshape(-1, 1)
                shift_y = (shift[:, 1]).reshape(-1, 1)

                shifted_x = x.reshape(-1,1) - shift_x 
                shifted_y = y.reshape(-1,1) - shift_y 

                shifted_coordinates = torch.cat((shifted_x.float(), shifted_y.float()), 1)
                trained_interpolation = torch.load('InterpNet.pt')
                shifted_f = trained_interpolation.forward(shifted_coordinates)

                if snap_counter == self.ref:
                    x_ref = x.clone().detach().numpy()
                    y_ref = y.clone().detach().numpy()
                    f_ref = f.clone().detach().numpy()

                elif snap_counter == self.test:
                    shift_x_test = shifted_x.clone().detach().numpy()
                    shift_y_test = shifted_y.clone().detach().numpy()
                    f_test = f.clone().detach().numpy()


                loss += torch.sum(torch.abs((shifted_f.flatten() - f)))
                snap_counter += 1

            loss = loss/Ns
            loss.backward()

            self.optimizer.step()

            with open('./Results/Loss.csv', 'a', newline='') as csvf:
                writer = csv.writer(csvf)
                writer.writerow([epoch, loss.item()])

            if epoch % 10 == 0:
                print('[Epoch {:4d}] {:18.8f}'.format(epoch, loss.item()))

            if epoch % 1 == 0:

                shift_plot(plot_counter, x_ref, y_ref, f_ref
                                       , shift_x_test, shift_y_test, f_test
                                       , loss.item())
                plot_counter += 1

            if epoch % 1000 == 0:
                with open('./Results/Training performance.txt', 'a') as f:
                    f.write("Epoch [{:d}]    ShiftNet loss = {:f}.\n".format(epoch, loss))

        self.save()


    def save(self):

        torch.save(self.model, 'ShiftNet.pt')








