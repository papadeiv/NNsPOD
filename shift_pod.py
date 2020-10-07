import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d

################################################################################
## Variables
################################################################################
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

func = nn.ReLU
lr = 0.00001
n_layers = 0
inner_size = 2
niter = 100
dt = 0.0784

################################################################################
## Prepare the dataset
################################################################################
timesteps = np.arange(1, 205) * dt
x_db = []
y_db = []
for i in range(1, 205):
    fname = 'data_wave_2/wave.{:d}.csv'.format(i)
    y, x = np.genfromtxt(fname, usecols=(0, 1), delimiter=',', skip_header=1).T
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
    nn.Linear(inner_size, 1), func()
)
trained_epoch = 0
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



################################################################################
## Training
################################################################################
for epoch in range(niter):
    '''
    '''
    def compute_loss():
        optimizer.zero_grad()

        loss = None

        x_ref = x_db.T[0].clone()
        y_ref = y_db.T[0].clone()
        for x, y, t in zip(x_db.T, y_db.T, timesteps):
            input_ = torch.cat((
                x.reshape(-1, 1).float(),
                torch.ones((x.shape[0], 1), dtype=torch.float, requires_grad=True)*t),
                axis=-1)

            shift = model(input_)
            new_x = x.reshape(shift.shape) - shift
            interpolator = interp1d(new_x.flatten().detach(), y.detach(), fill_value="extrapolate")
            shifted_snapshot = torch.tensor(interpolator(x.detach()))
            if loss is None:
                loss = criterion(shifted_snapshot, y_ref)
            else:
                f =  criterion(shifted_snapshot, y_ref)
                loss += f

        # Zero the gradients

        # perform a backward pass (backpropagation)
        loss.backward()
        print(loss, loss.grad, loss.requires_grad)
        return loss



    # Update the parameters
    loss = optimizer.step(compute_loss)

    if epoch % 2 == 0:
        print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  loss.item()))


