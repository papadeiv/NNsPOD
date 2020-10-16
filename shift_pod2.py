import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

################################################################################
## Variables
################################################################################
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

func = nn.Sigmoid
lr = 0.02
n_layers = 1
inner_size = 10
niter = 1000
dt = 0.0784

################################################################################
## Prepare the dataset
################################################################################
timesteps = np.arange(1, 205) * dt
x_db = []
y_db = []
for i in range(1, 205):
    #fname = 'data_wave_2/wave.{:d}.csv'.format(i)
    #y, x = np.genfromtxt(fname, usecols=(0, 1), delimiter=',', skip_header=1).T
    fname = 'data_wave_1/ffffff_{:d}.csv'.format(i)
    y, x = np.genfromtxt(fname, usecols=(1, 2), delimiter=',', skip_header=1).T
    y_db.append(y)
    x_db.append(x)
y_db = torch.tensor(y_db, requires_grad=True).T
x_db = torch.tensor(x_db, requires_grad=True).T
print(x_db.shape, y_db.shape)


################################################################################
## Build the network
################################################################################
inner_layers = []
for _ in range(n_layers):
    inner_layers.append(nn.Linear(inner_size, inner_size))
    inner_layers.append(func())

model = nn.Sequential(
    nn.Linear(2, inner_size), func(),
    *inner_layers,
    nn.Linear(inner_size, 1),
)
trained_epoch = 0
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def torch_interp1d(x, y, xs):
    ''' return ys, the interpolated values at `xs`'''
    pass
    #return ys

class InterpNet():

    def __init__(self, input, output):
        self.func = nn.Sigmoid
        self.lr = 0.1
        self.n_layers = 2
        self.inner_size = 20
        self.epoch = 1000
        inner_layers = []
        for _ in range(n_layers):
            inner_layers.append(nn.Linear(self.inner_size, self.inner_size))
            inner_layers.append(func())

        self.model = nn.Sequential(
            nn.Linear(1, self.inner_size), self.func(),
            *inner_layers,
            nn.Linear(self.inner_size, 1),
        )
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
            #lss.backward()
            self.optimizer.step()

    def forward(self, input):
        return self.model(input)


# x = torch.tensor([1, 2, 3])
# y = torch.tensor([5, 3, 10])
# xs = torch.linspace(1, 3, 128)
# ys = torch_interp1d(x, y, xs)
# print(ys)

def test_plot():


    idx_test = 0
    t = timesteps[idx_test]
    x_test = x_db.T[idx_test]
    y_test = y_db.T[idx_test]
    input_ = torch.cat((
        x_test.reshape(-1, 1).float(),
        torch.ones((x.shape[0], 1), dtype=torch.float, requires_grad=True)*t),
        axis=-1)

    shift = model(input_)
    print(shift.mean())
    new_x = x_test.reshape(shift.shape) - shift

    plt.cla()
    plt.plot(x_test.detach(), y_test.detach(), '-b', label='test before shift')
    plt.plot(new_x.detach(), y_test.detach(), '--r', label='test after shift')
    plt.plot(x_ref.detach(), y_ref.detach(), '--g', label='ref')
    plt.legend()
    plt.show()

x_ref = x_db.T[100].clone().float().reshape(-1, 1)
y_ref = y_db.T[100].clone().float().reshape(-1, 1)
print(x_ref.shape)
print(y_ref.shape)
true = torch.cat((x_ref.reshape(-1, 1), y_ref.reshape(-1, 1)), axis=-1)

interp1 = InterpNet(x_ref, y_ref)
'''
y_test = interp.forward(x_ref)
plt.plot(y_test.detach(), 'r-')
plt.plot(y_ref.detach(), 'b-')
plt.show()
'''

################################################################################
## Training
################################################################################
for epoch in range(niter):
    '''
    '''

    def compute_loss():
        optimizer.zero_grad()

        loss = 0.0
        id_snap = -1
        for x, y, t in zip(x_db.T, y_db.T, timesteps):
            id_snap += 1
            input_ = torch.cat((
                x.reshape(-1, 1).float(),
                torch.ones((x.shape[0], 1), dtype=torch.float, requires_grad=True)*t),
                axis=-1)

            shift = model(input_)
            new_x = x.reshape(-1, 1) - shift
            pred = interp1.forward(new_x.float()).flatten()
            loss += torch.sum(torch.abs((pred.flatten() - y)))

        # perform a backward pass (backpropagation)
        loss.backward()
        return loss



    # Update the parameters
    loss = optimizer.step(compute_loss)

    if epoch % 2 == 0:
        print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  loss.item()))

    if epoch % 100 == 0:
        test_plot()




