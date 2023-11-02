import pickle
import numpy as np
import torch
import os

def data_gen(Nx, Nt, range_x, range_t,mesh_type = 'linear',path = '', name = None):

    if not os.path.isdir(path):
        data_path = os.path.join(path, 'collocation_points/' + name)
        os.makedirs(data_path, exist_ok=True)

    if mesh_type == 'linear':
        interior_point, u_initial, u_boundary_left, u_boundary_right,collocation_points = mesh_2D(Nx, Nt, range_x, range_t)
    elif mesh_type == 'random':
        interior_point, u_initial, u_boundary_left, u_boundary_right,collocation_points = mesh_2D(Nx, Nt, range_x, range_t)

    path = path + 'collocation_points/'
    save_data(interior_point,name+'/interior_points',path)
    save_data(u_initial,name+'/initial_points',path)
    save_data(u_boundary_left,name+'/boundary_left_points',path)
    save_data(u_boundary_right,name+'/boundary_right_points',path)
    save_data(collocation_points,name+'/collocation_points',path)




def mesh_2D(Nx, Nt, range_x, range_t):

    x = torch.linspace(range_x[0], range_x[1], Nx, dtype=torch.double).view(-1)
    t = torch.linspace(range_t[0], range_t[1], Nt, dtype=torch.double).view(-1)

    collocation_points = torch.cartesian_prod(x, t)

    interior_point = torch.cartesian_prod(x[1: -1], t[1: -1])
    u_initial = torch.hstack((x.view(-1, 1), torch.zeros((Nx, 1))))
    'initial condition u(x,0)'

    u_boundary_left = torch.hstack((torch.full([Nt, 1], range_x[0]), t.view(-1, 1)))
    'left boundary condition u(0,t)'

    u_boundary_right = torch.hstack((torch.full([Nt, 1], range_x[1]), t.view(-1, 1)))
    'right boundary condition u(L,t)'

    return interior_point, u_initial, u_boundary_left, u_boundary_right,collocation_points


def random_mesh_2D(Nx, Nt, range_x, range_t):

    x = range_x[0] + (range_x[1] - range_x[0]) * torch.randn((Nx, 1),dtype =torch.double).view(-1)
    t = range_t[0] + (range_t[1] - range_t[0]) * torch.randn((Nt, 1),dtype =torch.double).view(-1)

    collocation_points = torch.cartesian_prod(x, t)

    interior_point = torch.cartesian_prod(x[1: -1], t[1: -1])
    u_initial = torch.hstack((x.view(-1, 1), torch.zeros((Nx, 1))))
    'initial condition u(x,0)'

    u_boundary_left = torch.hstack((torch.full([Nt, 1], range_x[0]), t.view(-1, 1)))
    'left boundary condition u(0,t)'

    u_boundary_right = torch.hstack((torch.full([Nt, 1], range_x[1]), t.view(-1, 1)))
    'right boundary condition u(L,t)'

    return [interior_point, u_initial, u_boundary_left, u_boundary_right, collocation_points]


def load_mesh(file_name,path):

    path = path + 'collocation_points/'
    interior_point =  load_data(file_name+'/interior_points',path)
    u_initial =  load_data(file_name+'/initial_points',path)
    u_boundary_left =  load_data(file_name+'/boundary_left_points',path)
    u_boundary_right =  load_data(file_name+'/boundary_right_points',path)
    collocation_points = load_data(file_name+'/collocation_points',path)

    return interior_point, u_initial, u_boundary_left, u_boundary_right, collocation_points



def load_data(name,path):
        with open(path + '/' + name + '.pkl','rb') as f:
            data =  pickle.load(f)
            f.close()
        return data

def save_data(data,name,path):
    with open(path + '/' + name + '.pkl','wb') as f:
        pickle.dump(data,f)
        f.close()

