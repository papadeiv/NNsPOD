from interp_net import InterpNet
from plot_test import shift_plot
import numpy as np
import torch
import torch.nn as nn
import math
import sys
import os
import csv

class ShiftNet():

    def __init__(self, ref, test):

        # ACTIVATION FUNCTION
        self.func = nn.PReLU
        # LEARNING RATE
        self.lr = 1e-5
        # N. HIDDEN LAYERS
        self.n_layers = 5
        # N. NEURONS IN HIDDEN LAYERS
        self.inner_size = 25
        # N. EPOCHS
        self.epoch = int(2e6)
        # ACCURACY TRESHOLD
        self.treshold = 1e0
        # LOSS FUNCTION
        self.criterion = torch.nn.L1Loss(reduction = 'sum')
        # DEFAULT (DO NOT CHANGE)
        self.checkpoint = 0
        self.plot_counter = 0
        self.ref = ref
        self.test = test

        inner_layers = []
        for _ in range(self.n_layers):
            inner_layers.append(nn.Linear(self.inner_size, self.inner_size))
            inner_layers.append(self.func())

        self.model = nn.Sequential(
            nn.Linear(3, self.inner_size), self.func(),
            *inner_layers,
            nn.Linear(self.inner_size, 2),)

        # LOSS OPTIMISATION ALGORITHM
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)


    def build_input(self, x, y, t):

        snap_t = torch.tensor(t*np.ones((x.shape[0], 1), dtype=float), requires_grad=True) # snap_t.shape = torch.Size([Nh, 1])
        snap_x = x.reshape(-1, 1)
        snap_y = y.reshape(-1, 1)

        coordinates = torch.cat((snap_x.float(), snap_y.float(), snap_t.float()), 1)

        return coordinates


    def forward(self, coordinates):

        return self.model(coordinates)


    def train(self, X, Y, Ts, F):

        Ns = Ts.size

        if os.path.exists('./TrainedModels/ShiftNet.pt'):

            print('Pre-trained model found\nLoading state parameters\n\n')
            
            self.load()
            for name, param in self.model.named_parameters():
                print(name, param)

        else:
            with open('./Results/ShiftNetLoss.csv', 'w+', newline='') as csvf:
                writer = csv.writer(csvf)
                writer.writerow(['Epoch', 'Loss'])

        trained_interpolation = torch.load('./TrainedModels/InterpNet.pt')

        ## Extract reference and test torch tensors from the dataset ## 
        tensor_f_ref = F[:,self.ref].clone().float().reshape(-1, 1)
        tensor_x_ref = X[:,self.ref].clone().float().reshape(-1, 1)
        tensor_y_ref = Y[:,self.ref].clone().float().reshape(-1, 1)
        tensor_f_test = F[:,self.test].clone().float().reshape(-1, 1)
        tensor_x_test = X[:,self.test].clone().float().reshape(-1, 1)
        tensor_y_test = Y[:,self.test].clone().float().reshape(-1, 1)

        ## Build associated numpy vectors for shift_plot function ##
        f_ref = tensor_f_ref.clone().detach().numpy()
        x_ref = tensor_x_ref.clone().detach().numpy()
        y_ref = tensor_y_ref.clone().detach().numpy()
        f_test = tensor_f_test.clone().detach().numpy()

        for epoch in range(self.checkpoint, self.epoch):

            ## Variable that contains the order of magnitude of the epoch to limit I/O operations ##
            magnitude = pow(10, math.floor(math.log10(epoch+1)))

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
                shifted_f = trained_interpolation.forward(shifted_coordinates)

                if snap_counter == self.test:
                    shift_x_test = shifted_x.clone().detach().numpy()
                    shift_y_test = shifted_y.clone().detach().numpy()

                loss += self.criterion(shifted_f.flatten(), f)
                snap_counter += 1

            loss = loss/Ns
            loss.backward()

            self.optimizer.step()
            self.save(epoch)

            with open('./Results/ShiftNetLoss.csv', 'a', newline='') as csvf:
                writer = csv.writer(csvf)
                writer.writerow([epoch, loss.item()])

            if epoch == 0:

                shift_plot(self.plot_counter, epoch, x_ref, y_ref, f_ref
                                       , shift_x_test, shift_y_test, f_test
                                       , loss.item())
                self.plot_counter += 1

            if epoch % 10 == 0:

                print('[Epoch {:4d}] {:18.8f}'.format(epoch, loss.item()))

            if epoch % 100 == 0 and epoch != 0:

                print('\n### Checkpoint: saving the model state (weights and biases only) for resuming training ###\n')
                self.save(epoch)

            if epoch % magnitude == 0 and epoch >= 10:

                shift_plot(self.plot_counter, epoch, x_ref, y_ref, f_ref
                                       , shift_x_test, shift_y_test, f_test
                                       , loss.item())
                self.plot_counter += 1


            if loss.item() < self.treshold:

                print("\n**** ShiftNet's accuracy treshold reached ****\n")
                shift_plot(self.plot_counter, epoch, x_ref, y_ref, f_ref
                                       , shift_x_test, shift_y_test, f_test
                                       , loss.item())

                break


        print('Training completed: saving the full model (weights, biases and architecture) for generalisation \n\n')
        torch.save(self.model, './TrainedModels/ShiftNet.pt')


    def save(self, epoch):

        script_dir = os.path.dirname(__file__)
        res_dir = os.path.join(script_dir, 'TrainedModels/')
        os.makedirs(res_dir, exist_ok=True)

        state = {
            'epoch': epoch+1,
            'plot_counter': self.plot_counter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}

        torch.save(state, './TrainedModels/ShiftNet.pt')

    def load(self):

        state = torch.load('./TrainedModels/ShiftNet.pt')

        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.checkpoint = state['epoch']
        self.plot_counter = state['plot_counter']










