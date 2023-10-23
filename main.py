import torch
import matplotlib.pyplot as plt
import numpy as np
from training import train
from samples_gen import *
from functions import *
from plotting import *
from PDEs import Pde
from NN import *


def main():

    n_layers,n_neurons = 2,128
    net = FNN(n_layers,n_neurons).cuda()
    Nx = 100
    Nt = 100
    range_x = [0,1]
    range_t = [0,1]
    iteration = 10000
    learning_rate = 1e-3

    #transport equation
    t_equation = lambda x: np.cos(2*np.pi * (x[0]-2 * x[1]))
    '''
    u = cos(2pi(x-2t))
    ut + 2ux = 0
    x,t in [0,1]^2
    '''
    transport_equation = Pde(2, [[0,1],[0,1]], t_equation, time_dependency = True)
    mesh = mesh_2D(Nx, Nt, range_x, range_t)


    train(transport_equation,net,mesh,lr = learning_rate  ,iteration = iteration,save=True)

if __name__ == '__main__':
    main()