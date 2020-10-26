import torch
import torch.nn as nn
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('WebAgg')
import sys
import os


###########################################################################################
################################## POST-PROCESSING TOOLS ##################################
###########################################################################################


def zero_to_nan(array):

    return [float('nan') if x<=0.01 else x for x in array]

def interpTestPlot(model,idx,epoch,loss):

    t_test = torch.tensor(timesteps[test]*np.ones((Nh, 1), dtype=float), requires_grad=True)
    x_test = (x[:,test].clone()).reshape(-1, 1)
    y_test = (y[:,test].clone()).reshape(-1, 1)
    f_test = (f[:,test].clone()).reshape(-1, 1)

    input_ = torch.cat((x_test, y_test), 1)
    interpolated_f_test = model(input_.float())

    X, Y = np.meshgrid(xMesh, yMesh)
    fMesh = interpolated_f_test.reshape(50,50).detach()

    contour_fig = plt.figure()
    ax = contour_fig.gca(projection='3d')

    ax.contour(X, Y, fMesh, colors='black')
    ax.contour(X, Y, f_refMesh, colors='green')

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.view_init(elev=90, azim=270)

    plt.suptitle('Epoch [{:d}]: Interpolation loss = {:f}\nReference (75) snapshot = green; Interpolated test (0) snapshot = black'.format(epoch, loss), fontsize=8)
    plt.savefig('Interpcontour{:d}.png'.format(idx))

def multiPlot(idx,epoch,loss):
    
    t_test = torch.tensor(timesteps[test]*np.ones((Nh, 1), dtype=float), requires_grad=True)
    x_test = (x[:,test].clone()).reshape(-1, 1)
    y_test = (y[:,test].clone()).reshape(-1, 1)
    f_test = (f[:,test].clone()).reshape(-1, 1)
    
    ################################################
    ################## TEST SHIFT ##################
    ################################################

    input_ = torch.cat((x_test.float(), y_test.float(), t_test.float()), 1)

    shift = ShiftNet(input_)
    shift_x = (shift[:, 0]).reshape(-1, 1)
    shift_y = (shift[:, 1]).reshape(-1, 1)

    shifted_x = x_test - shift_x
    shifted_y = y_test - shift_y
    shifted_y_mesh = shifted_y.reshape(50,50)

    ###################################################################
    ########################## CONTOUR PLOT  ##########################
    ###################################################################

    shifted_xMesh = np.array(shifted_x[:50].detach())
    shifted_yMesh = np.array(shifted_y_mesh[:,1].detach())

    X, Y = np.meshgrid(xMesh, yMesh)
    shiftX, shiftY = np.meshgrid(shifted_xMesh, shifted_yMesh)

    input_ = torch.cat((x_test, y_test), 1)
    interpolated_f_test = trained_interpolation.forward(input_.float())
    fMesh = interpolated_f_test.reshape(50,50).detach()

    contour_fig = plt.figure()
    ax1 = contour_fig.gca(projection='3d')

    ax1.contour(X, Y, f_testMesh)
    ax1.contour(shiftX, shiftY, f_testMesh, cmap=cm.coolwarm)
    ax1.contour(X, Y, f_refMesh, colors='green')

    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.view_init(elev=90, azim=270)

    plt.suptitle('Epoch [{:d}]: Interpolation loss = {:f}\nReference (green) = 75; Test (linear YGB gradient) = 0; Shifted-test (cool-warm gradient).'.format(epoch,loss), fontsize=8)
    plt.savefig('contour{:d}.png'.format(idx))

    ################################################################
    ########################## SURF PLOT  ##########################
    ################################################################

    trisurf_fig = plt.figure()
    ax2 = trisurf_fig.add_subplot(111, projection='3d')

    ax2.plot_trisurf(x_test.detach().flatten(), y_test.detach().flatten(), zero_to_nan(f_test.detach().flatten()), color="b", label="Test")
    ax2.plot_trisurf(shifted_x.detach().flatten(), shifted_y.detach().flatten(), zero_to_nan(f_test.detach().flatten()), color="r", label="Shifted-test")  
    ax2.plot_trisurf(x_ref.detach().flatten(), y_ref.detach().flatten(), zero_to_nan(f_ref.detach().flatten()), color="g", label="Reference")

    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('y', fontsize=10)
    ax2.view_init(elev=90, azim=270)
    
    plt.suptitle('Epoch [{:d}]: Interpolation loss = {:f}\nReference (Green) = 75; Test (Blue) = 0; Shifted-test (Red).'.format(epoch, loss), fontsize=8)
    plt.savefig('trisurf{:d}.png'.format(idx))


#############################################################################################
################################## ASSEMBLY OF THE DATASET ##################################
#############################################################################################


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

xMesh = np.array(x[ref][:50])
yMesh = xMesh

f_refMesh = np.array(zero_to_nan(f[ref]))
f_refMesh = f_refMesh.reshape(50,50)
f_testMesh = np.array(zero_to_nan(f[test]))
f_testMesh = f_testMesh.reshape(50,50)


f = torch.tensor(np.array(f), requires_grad=True).T # f.shape = torch.Size([Nh, Ns])
x = torch.tensor(np.array(x), requires_grad=True).T # x.shape = torch.Size([Nh, Ns])
y = torch.tensor(np.array(y), requires_grad=True).T # y.shape = torch.Size([Nh, Ns])

fio.write("\nNumber of snapshots(Ns): {:d}\nFull-order (snapshot) dimension(Nh): {:d}\n".format(Ns, Nh))
print("Number of snapshots(Ns): {:d}\nFull-order dimension(Nh): {:d}\n".format(Ns, Nh))


##############################################################################################
################################## INTERPOLATION NEURAL NET ##################################
##############################################################################################


class InterpNet():

    def __init__(self, input, output):

        self.func = nn.Sigmoid
        self.lr = 0.001
        self.n_layers = 2
        self.inner_size = 40
        self.epoch = 20000

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

        plt_counter = np.int(0)

        for epoch in range(self.epoch):
            y_pred = self.model(input)
            lss = self.criterion(output, y_pred)

            if epoch == 0:
                fio.write("\nInterpNet loss starts at {:f}.".format(lss.item()))

            if epoch % 100 == 0:
                print('[epoch {:4d}] {:18.8f}'.format(
                    epoch,  lss.item()))

                plt_counter += 1

                interpTestPlot(self.model, plt_counter, epoch, lss.item())

            self.model.zero_grad()
            lss.backward(retain_graph=True)
            self.optimizer.step()

        fio.write("\nInterpNet loss after {:d} epochs: {:f}\n.".format(epoch, lss.item()))

    def forward(self, input):
        return self.model(input)


f_ref = f[:,ref].clone().float().reshape(-1, 1) # f_ref.shape = torch.Size([Nh, 1])
x_ref = x[:,ref].clone().float().reshape(-1, 1) # x_ref.shape = torch.Size([Nh, 1])
y_ref = y[:,ref].clone().float().reshape(-1, 1) # y_ref.shape = torch.Size([Nh, 1])

input_ref = torch.cat((x_ref, y_ref), 1) # input_ref.shape = torch.Size([Nh, 2])
# np.savetxt("input_ref", input_ref.detach().numpy(), newline="\n")

trained_interpolation = InterpNet(input_ref, f_ref)


################################################################################################
################################## SHIFT-DETECTING NEURAL NET ##################################
################################################################################################


dyn_lr = 0.1
n_layers = 3
inner_size = 20
n_epochs = 1000


inner_layers = []
for _ in range(n_layers):
    inner_layers.append(nn.Linear(inner_size, inner_size))
    inner_layers.append(nn.Sigmoid())

ShiftNet = nn.Sequential(
    nn.Linear(3, inner_size), nn.Sigmoid(),
    *inner_layers,
    nn.Linear(inner_size, 2),)

optimizer = torch.optim.Adam(ShiftNet.parameters(), lr=dyn_lr)

previous_loss = sys.float_info.max
counter_lr = 0
plt_counter = 0

for epoch in range(n_epochs):

    def compute_loss():

        optimizer.zero_grad()
        loss = 0.0

        for snap_x, snap_y, t, snap_f in zip(x.T, y.T, timesteps, f.T):

            #np.savetxt("x[epoch={:d}, timestep={:d}".format(epoch, idx_t), j.detach(), newline="\n")

            snap_t = torch.tensor(t*np.ones((Nh, 1), dtype=float), requires_grad=True) # snap_t.shape = torch.Size([Nh, 1])
            snap_x = snap_x.reshape(-1, 1) # snap_x.shape = torch.Size([Nh, 1])
            snap_y = snap_y.reshape(-1, 1) # snap_y.shape = torch.Size([Nh, 1])

            input_ = torch.cat((snap_x.float(), snap_y.float(), snap_t.float()), 1) # input_.shape = torch.Size([Nh, 3])
            # np.savetxt("input_", input_.detach().numpy(), newline="\n")

            shift = ShiftNet(input_) # shift.shape = torch.Size([Nh, 2])
            shift_x = (shift[:, 0]).reshape(-1, 1)
            shift_y = (shift[:, 1]).reshape(-1, 1)

            shifted_x = snap_x - shift_x # shifted_x.shape = torch.Size([Nh, 1])
            shifted_y = snap_y - shift_y # shifted_y.shape = torch.Size([Nh, 1])
            shifted_input = torch.cat((shifted_x, shifted_y), 1) # shifted_input.shape = torch.Size([Nh, 2])
            # np.savetxt("shifted_input", shifted_input.detach().numpy(), newline="\n")

            shifted_f = trained_interpolation.forward(shifted_input.float()) # shifted_f.shape = torch.Size([Nh, 1])
            # np.savetxt("shifted_f", shifted_f.detach(), newline="\n")

            loss += torch.sum(torch.abs((shifted_f.flatten() - snap_f)))

        loss = loss/Ns
        loss.backward()
        return loss


    loss = optimizer.step(compute_loss)

    if (loss - previous_loss >= np.finfo(float).eps**4 and dyn_lr >= 0.00000001):

        counter_lr += 1
        
        fio.write("     Learning rate updated from {:f}.\n".format(dyn_lr))
        dyn_lr = 0.75*dyn_lr
        fio.write("to {:f}".format(dyn_lr))

        for g in optimizer.param_groups:

            g['lr'] = dyn_lr

        print("     Learning rate update: {:f}.".format(optimizer.param_groups[0]['lr']))


    previous_loss = loss

    if epoch % 10 == 0:

        print('[Epoch {:4d}] {:18.8f}'.format(epoch, loss.item()))
        fio.write("Epoch [{:d}]    ShiftNet loss = {:f}.\n".format(epoch, loss))

    if epoch <=50:

        plt_counter += 1

        multiPlot(plt_counter,epoch,loss.item())

    elif epoch % 10 == 0:

        plt_counter += 1

        multiPlot(plt_counter,epoch,loss.item())



fio.write("\n\nThe learning rate has been updated {:d} times".format(counter_lr))
fio.close()


###############################################################################################
################################## GENERALIZATION AND OUTPUT ##################################
###############################################################################################


X = np.zeros((Nh, Ns))
counter = 0

for j, k, t in zip(x.T, y.T, timesteps):

    input_ = torch.cat((j.reshape(-1, 1).float(), k.reshape(-1, 1).float(), torch.ones((j.shape[0],1), dtype=torch.float, requires_grad=True)*t.item()), axis=-1)

    shift = ShiftNet(input_)

    shifted_x = j.reshape(-1, 1) - shift[:, 0].reshape(-1, 1)
    shifted_y = k.reshape(-1, 1) - shift[:, 1].reshape(-1, 1)

    shifted_input = torch.cat((shifted_x.reshape(-1, 1), shifted_y.reshape(-1, 1)), 1)

    shifted_f = trained_interpolation.forward(shifted_input.float())

    shifted_f = shifted_f.flatten()

    snapshot = shifted_f.clone()

    snapshot = snapshot.detach().numpy()

    X[:, counter] = snapshot

    counter+=1

np.save("shifted_snapshot_matrix({:d})".format(ref), X)