from torch import optim
from torch.autograd import Variable
from NN import *
import os

def train(pde,net,mesh_point,lr=1e-3, iteration = 5000, save = True, model_name = 'Model.pt', CUDA = True, path = ''):

    if CUDA == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            print('No cuda can be used')
        print(device)

    else:
        device = 'cpu'
        print(device)


    collocation_points, x, u_initial, u_boundary_left, u_boundary_right = mesh_point



    u_initial_y = torch.from_numpy(pde.solutions(u_initial)).float().to(device)
    u_boundary_left_y = torch.from_numpy(pde.solutions(u_boundary_left)).float().to(device)
    u_boundary_right_y = torch.from_numpy(pde.solutions(u_boundary_right)).float().to(device)


    collocation_points = torch.from_numpy(collocation_points).float().to(device)
    u_initial = torch.from_numpy(u_initial).float().to(device)
    u_boundary_left = torch.from_numpy(u_boundary_left).float().to(device)
    u_boundary_right = torch.from_numpy(u_boundary_right).float().to(device)


    size = len(collocation_points)

    optimizer = optim.Adam(net.parameters(),lr)
    #optimizer = optim.SGD(net.parameters(),lr)
    loss_fun = nn.MSELoss()
    '''
    define optimizer and loss function
    '''

    total_loss = []
    loss_on_pde = []
    loss_on_boundary = []



    # training
    for i in range(iteration):
        collocation_points = Variable(collocation_points, requires_grad=True)


        du = torch.autograd.grad(net(collocation_points), collocation_points, grad_outputs=torch.ones_like(net(collocation_points)), create_graph=True)
        ux = du[0][:, 0].unsqueeze(-1).to(device)
        ut = du[0][:, 1].unsqueeze(-1).to(device)


        optimizer.zero_grad()


        # loss function:
        loss1 = loss_fun(net(u_initial), u_initial_y)
        # computing the loss of u(x,0)


        loss2 = loss_fun(net(u_boundary_left),u_boundary_left_y)
        # computing the loss of u(0,t)


        loss3 = loss_fun(net(u_boundary_right),net(u_boundary_left))
        # computing the loss of u(L,t)


        loss4 = loss_fun(ut + 2*ux, torch.zeros([size, 1]).to(device))
        # computing the loss of u_t = u_xx


        loss = loss1+loss2+loss3+loss4
        pde_loss = loss4
        boundary_loss = loss1+loss2+loss3


        loss.backward()
        optimizer.step()


        total_loss.append(loss.item())
        loss_on_pde.append(pde_loss.item())
        loss_on_boundary.append(boundary_loss.item())

        if i % 1000 == 0:
            print(
                f'total loss = {loss.item()}     pde_loss = {pde_loss.item()}      boundary_loss = {boundary_loss.item()}')

        if loss.item() <= 1e-4:
            return



    if save == True:
        if not os.path.isdir(path):
            model_path = os.path.join(path, 'model')
            os.makedirs(model_path,exist_ok = True)

        torch.save(net, f"model/{model_name}")
        data = {'total': total_loss, 'pde': loss_on_pde, 'boundary': loss_on_boundary}



