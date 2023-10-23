from  NN import *
import numpy as np



def load_model(configure,device = 'cpu',name = 'Model.pt'):
    if configure['act'] == 'tanh':
        test_net = PINN_Tanh(configure['structure'][0], configure['structure'][1])
    elif configure['act'] == 'sigmoid':
        test_net = PINN_Sig(configure['structure'][0], configure['structure'][1])

    test_net = torch.load(f"model/{name}", map_location=torch.device(device))
    test_net.eval()

    return  test_net

def configure(structure, act, range_x, range_t, Num_x, Num_t,mesh_type):
    info = {}
    info['structure'] = structure
    info['act'] = act
    info['x_range'] = range_x
    info['t_range'] = range_t
    info['Num_x'] = Num_x
    info['Num_t'] = Num_t
    info['type_mesh'] = mesh_type
    info['name'] = f"{mesh_type}_{structure}_{Num_x}x{Num_t}_{act}.pt"
    return info


def test_configure(structure, act, range_x, range_t, Num_x, Num_t,mesh_type): #####test
    info = {}
    info['structure'] = structure
    info['act'] = act
    info['x_range'] = range_x
    info['t_range'] = range_t
    info['Num_x'] = Num_x
    info['Num_t'] = Num_t
    info['type_mesh'] = mesh_type
    info['name'] = f"{mesh_type}_{structure}_{Num_x}x{Num_t}_{act}.pt"
    return info