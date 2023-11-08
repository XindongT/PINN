import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import seaborn as sns
from matplotlib.ticker import LinearLocator

def heat_map(net,N_x,N_y,x_range,y_range,device = 'cpu',cmap = None):
    net.eval()
    x = torch.linspace(x_range[0],x_range[1],N_x)
    y = torch.linspace(y_range[0],y_range[1],N_y)

    X = torch.stack(torch.meshgrid(x,y)).view(2,-1).T
    X = X.to(device)
    with torch.no_grad():
        pre = net(X).view(len(x),len(y)).cpu().numpy()
    sns.heatmap(pre,cbar=True,cmap = cmap)
    plt.show()

def plot_solution_2D(pde,t,N_x,N_y,x_range,y_range):
    x = np.linspace(x_range[0],x_range[1],N_x)
    y = np.linspace(y_range[0],y_range[1],N_y)

    X = np.stack(np.meshgrid(x,y)).reshape(2,-1).T
    pre = pde.solutions(X).reshape(len(x),len(y))

    plt.plot(pre.T[t])
    plt.show()

def plot_solution_3D(pde,N_x,N_y,x_range,y_range):
    x = torch.linspace(x_range[0],x_range[1],N_x)
    y = torch.linspace(y_range[0],y_range[1],N_y)

    mesh_x,mesh_y = np.meshgrid(x,y)
    X = np.stack((mesh_x,mesh_y)).reshape(2,-1).T

    pre = pde.solutions(X).reshape(len(x),len(y))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mesh_x, mesh_y, pre, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_2D(net,t,N_x,N_y,x_range,y_range,device = 'cpu'):
    net.eval()
    x = torch.linspace(x_range[0],x_range[1],N_x)
    y = torch.linspace(y_range[0],y_range[1],N_y)

    X = torch.stack(torch.meshgrid(x,y)).view(2,-1).T
    X = X.to(device)
    with torch.no_grad():
        pre = net(X).view(len(x),len(y)).cpu().numpy()

    X = X.cpu().numpy()

    plt.plot(pre.T[t])
    plt.show()


def plot_3D(net,N_x,N_y,x_range,y_range,device = 'cpu'):
    net.eval()
    x = torch.linspace(x_range[0],x_range[1],N_x)
    y = torch.linspace(y_range[0],y_range[1],N_y)

    mesh_x,mesh_y = torch.meshgrid(x,y)
    X = torch.stack((mesh_x,mesh_y)).view(2,-1).T
    X = X.to(device)
    with torch.no_grad():
        pre = net(X).view(len(x),len(y)).cpu().numpy()


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mesh_x, mesh_y, pre, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()



