import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import sys

################################################################################
## Variables
################################################################################
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

actv = nn.ReLU
sigm = nn.Sigmoid
lr = 0.01
n_layers = 2
inner_size = 2
niter = 2
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
    inner_layers.append(actv())

model = nn.Sequential(
    nn.Linear(2, inner_size), actv(),
    nn.Linear(inner_size, inner_size), actv(),
    nn.Linear(inner_size, inner_size), actv(),
    nn.Linear(inner_size, 1), sigm()
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

        y_ref = y_db.T[0].clone()

        for n in range(len(timesteps)):

            t = timesteps[n]

            # Consider the n-th snapshot

            x = x_db.T[n]
            x_ref = x.detach().numpy()
            x_copy = x

            # print("We are considering the ", n, "-th snapshot")
            # print("Pre-shift centroids: ", x)

            # Initialize a shift vector to be filled for each snapshot

            shift = []

            for j in range(len(x)):

                # Forward propagation pass on the net

                output = model(torch.tensor([x[j], t], dtype=torch.float))

                # Build the previously initialised shift vector of which the mean value will be taken

                shift.append(output)

                # sys.exit(j, n)
            
            # Compute the mean of the shift vector

            pred_shift_stretch = torch.mean(torch.tensor(shift))

            # print(pred_shift_stretch)

            # Update the coordinates vector with the new shifted (stretched) values

            x = x*pred_shift_stretch

            # print("Pre-interpolation shifted centroids: ", x)

            def find_closest_centroids(x):

                # Initialize a list of 2-dimensional closest centroids coordinates
                # in which each entry is [x_min, x_max]

                closest_centroids = []
                index_list = []
                distances = []

                for j in range(len(x)):

                    # Calculate the closest centroid in the L1-norm

                    x_min = (torch.abs(x_copy - x[j])).argmin()

                    # Find the second centroids that defines the interval [x_min, x_max]
                    # in which the shifted centroid ended up

                    if(x[j]-x_copy[x_min]>0):
                        x_max = x_min + 1
                    elif(x[j]-x_copy[x_min]<0):
                        x_max = x_min
                        x_min = x_min - 1
                    else:
                        x_max = x_min

                    # Calculate the distances between the shifted coordinate and its
                    # closest centroids

                    dx_min = x[j] - x_copy[x_min]
                    dx_max = x_copy[x_max] - x[j]

                    # Update the list of closest centroids

                    closest_centroids.append([x_ref[x_min], x_ref[x_max]])
                    index_list.append([x_min, x_max])
                    distances.append([dx_min, dx_max])

                return closest_centroids, index_list, distances

            new_x, new_x_list, dx = find_closest_centroids(x)

            # print("Post-interpolation shifted centroids: ", torch.tensor(new_x, requires_grad=True))

            # Extract the n-th snapshot

            y = y_db.T[n]

            # print("Pre-shift and pre-interpolation snapshot's values: ", y)

            # Interpolate linearly the value of the function in the shifted centroid
            # according to its value onto its closest centroids

            shifted_y = torch.zeros(len(new_x))

            interp_coeff = 0.5

            for j in range(len(new_x)):

                y_to_be_shifted = (y[j]).detach().numpy()

                target_x_min = np.int8(new_x_list[j][0])
                target_x_max = np.int8(new_x_list[j][1])

                dx_min_from_target = np.float32((dx[j][0]).detach().numpy())
                dx_max_from_target = np.float32((dx[j][1]).detach().numpy())

                shifted_y[target_x_min] += -y_to_be_shifted*interp_coeff*dx_min_from_target
                shifted_y[target_x_max] += y_to_be_shifted*interp_coeff*dx_max_from_target

            # print("Post-shift and post-interpolation snapshot's values: ", shifted_y)

            # shifted_y = torch.tensor(shifted_y, dtype=torch.double, requires_grad=True)

            shifted_y = shifted_y.clone().detach().requires_grad_(True)

            if loss is None:
                loss = criterion(y, shifted_y)
                loss.retain_grad()
            else:
                loss += criterion(y, shifted_y)
                loss.retain_grad()

        # Backward propagation pass on net

        # loss.retain_grad()
        loss.backward()
        print(loss, loss.grad)
        return loss


    # Update the parameters
    loss = optimizer.step(compute_loss)

    # if epoch % 2 == 0:
    #     print('[epoch {:4d}] {:18.8f}'.format(
    #                 epoch,  loss.item()))


