import pickle
import torch
import os

def load_data(name, path):
    with open(path + '/' + name + '.pkl', 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data


def save_data(data, name, path):
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(data, f)
        f.close()


def load_model(model_name='Model', model_path='', device='cpu'):
    path = os.path.join(model_path, model_name)
    return torch.load(path, map_location=device)


def load_mesh(file_name,path):

    path = path + 'collocation_points/'
    interior_point =  load_data(file_name+'/interior_points',path)
    u_initial =  load_data(file_name+'/initial_points',path)
    u_boundary_left =  load_data(file_name+'/boundary_left_points',path)
    u_boundary_right =  load_data(file_name+'/boundary_right_points',path)
    collocation_points = load_data(file_name+'/collocation_points',path)

    return interior_point, u_initial, u_boundary_left, u_boundary_right, collocation_points