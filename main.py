from torch import optim,nn
from training import train
from samples_gen import *
from PDEs import Pde
from data_sets import *
from NN import FNN
import argparse

def main():

    n_layers,n_neurons = 2,128
    net = FNN(n_layers,n_neurons).cuda()
    Nx = 100
    Nt = 100
    range_x = [0,1]
    range_t = [0,1]
    iteration = 2000
    learning_rate = 1e-3

    working_dir = ''
    collocation_file = 'linear_mesh'


    #transport equation
    solution = lambda x: np.cos(2*np.pi * (x[0]-2 * x[1]))

    '''
    u = cos(2pi(x-2t))
    ut + 2ux = 0
    x,t in [0,1]^2
    '''

    data_gen(Nx, Nt, range_x, range_t, mesh_type='linear', path=working_dir, name=collocation_file)
    transport_equation = Pde(2, [[0,1],[0,1]], solution, time_dependency = True)

    optimizer = optim.Adam(net.parameters(), learning_rate)
    loss_fun = nn.MSELoss()

    train(net, optimizer, loss_fun, transport_equation, iteration = iteration, save = True, path = working_dir, file_name = collocation_file)

if __name__ == '__main__':
    main()