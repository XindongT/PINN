import numpy as np



def Transport(x,y):
    return np.sin(y-x**2/2)
'''
x,y in R
u_x + xu_y = 0
u(0,y) = sin(y)
'''



def RR_sample(Num_x, Num_t, range_x, range_t):
    sample_size = Num_t * Num_x
    x = range_x[0] + (range_x[1] - range_x[0]) * np.random.rand(sample_size, 1)
    t = range_t[0] + (range_t[1] - range_t[0]) * np.random.rand(sample_size, 1)

    x_train = np.hstack((x, t))

    u_initial = np.hstack((x, np.zeros((sample_size, 1))))

    u_boundary_left = np.hstack((np.full([sample_size, 1], range_x[0]), t))

    u_boundary_right = np.hstack((np.full([sample_size, 1], range_x[1]), t))

    return x_train, x, u_initial, u_boundary_left, u_boundary_right


def mesh_2D(Nx, Nt, range_x, range_t):
    x = np.linspace(range_x[0], range_x[1], Nx).reshape(-1, 1)
    t = np.linspace(range_t[0], range_t[1], Nt).reshape(-1, 1)

    x_mesh, t_mesh = np.meshgrid(x[1:-1], t[1:])
    x_mesh = x_mesh.reshape(-1,1)
    t_mesh = t_mesh.reshape(-1,1)

    interior_point = np.hstack(((x_mesh, t_mesh)))
    u_initial = np.hstack((x, np.zeros((Nt,1))))
    'initial condition u(x,0)'

    u_boundary_left = np.hstack((np.full([Nt,1], range_x[0]), t.reshape(-1,1)))
    'left boundary condition u(0,t)'

    u_boundary_right = np.hstack((np.full([Nt,1], range_x[1]), t.reshape(-1,1)))
    'right boundary condition u(L,t)'

    return [interior_point, x, u_initial, u_boundary_left, u_boundary_right]




