import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from scipy.interpolate import interp1d

################################################################################
## Variables
################################################################################
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

func = nn.Sigmoid
lr = 0.001
n_layers = 3
inner_size = 40
niter = 1000
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
    nn.Linear(inner_size, 1),
)
trained_epoch = 0
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def torch_interp1d(x, y, xs):
    ''' return ys, the interpolated values at `xs`'''
    pass
    #return ys

# x = torch.tensor([1, 2, 3])
# y = torch.tensor([5, 3, 10])
# xs = torch.linspace(1, 3, 128)
# ys = torch_interp1d(x, y, xs)
# print(ys)

def test_plot():
    import matplotlib
    matplotlib.use('WebAgg')
    import matplotlib.pyplot as plt


    idx_test = 100
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
    plt.plot(x_db.T[0].detach(), y_db.T[0].detach(), '--g', label='ref')
    plt.legend()
    plt.show()

################################################################################
## Training
################################################################################
for epoch in range(niter):
    '''
    '''
    x_ref = x_db.T[0].clone()
    y_ref = y_db.T[0].clone()
    true = torch.cat((x_ref.reshape(-1, 1), y_ref.reshape(-1, 1)), axis=-1)

    def compute_loss():
        optimizer.zero_grad()

        loss = 0.0

        for x, y, t in zip(x_db.T, y_db.T, timesteps):
            input_ = torch.cat((
                x.reshape(-1, 1).float(),
                torch.ones((x.shape[0], 1), dtype=torch.float, requires_grad=True)*t),
                axis=-1)

            shift = model(input_)
            new_x = x.reshape(-1, 1) - shift
            pred = torch.cat((new_x, y.reshape(-1, 1)), axis=-1)
            mask = torch.nonzero(true[:, 1]).flatten()
            maskk = torch.nonzero(pred[:, 1]).flatten()[:mask.shape[0]]
            loss += criterion(pred[maskk], true[mask])

            #loss += torch.mean(torch.sum((pred-true)**2, 1))
            #print(loss)


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




