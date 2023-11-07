from torch.autograd import Variable
from NN import *
import os
import copy
from utils import load_mesh,save_data


def train(net, optimizer, loss_fun, pde, iteration = 5000, validation_set = None ,save = True, model_name = 'Model', CUDA = True, path = '', file_name = None):

    if CUDA == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            print('No cuda can be used')
        print(device)

    else:
        device = 'cpu'
        print(device)

    if file_name == None:
        raise ValueError('Please generate collocation map before training')


    interior_points, u_initial, u_boundary_left, u_boundary_right, collocation_points = load_mesh(file_name, path)

    u_initial_y = torch.from_numpy(pde.solutions(u_initial)).float().to(device).unsqueeze(-1)
    u_boundary_left_y = torch.from_numpy(pde.solutions(u_boundary_left)).float().to(device).unsqueeze(-1)
    u_boundary_right_y = torch.from_numpy(pde.solutions(u_boundary_right)).float().to(device).unsqueeze(-1)
    RHS = torch.zeros([len(collocation_points), 1],device = device)
    collocation_points_y = torch.from_numpy(pde.solutions(collocation_points)).float().to(device).unsqueeze(-1)

    interior_points = interior_points.float().to(device)
    u_initial = u_initial.float().to(device)
    u_boundary_left = u_boundary_left.float().to(device)
    u_boundary_right = u_boundary_right.float().to(device)
    collocation_points = collocation_points.float().to(device)


    total_loss = []
    loss_on_pde = []
    loss_on_boundary = []
    loss_on_initial = []

    best_epoch = None
    best_model = None

    min_val_error = float(99999)

    # training
    if validation_set is not None:
        interior_points_val, u_initial_val, u_boundary_left_val, u_boundary_right_val,collocation_points_val = validation_set
        collocation_points_val_y = collocation_points

    for i in range(iteration):
        # interior_points = Variable(interior_points, requires_grad=True)
        # u_boundary_left = Variable(u_boundary_left, requires_grad = True)
        # u_boundary_right = Variable(u_boundary_right, requires_grad = True)
        # u_initial = Variable(u_initial, requires_grad = True)
        collocation_points = Variable(collocation_points,requires_grad = True)

        du = torch.autograd.grad(inputs = collocation_points, outputs = net(collocation_points), grad_outputs=torch.ones_like(net(collocation_points)), create_graph=True)[0]
        ux = du[:, 0].unsqueeze(-1).to(device)
        ut = du[:, 1].unsqueeze(-1).to(device)
        LHS = ut + 2 * ux

        # loss function:
        loss1 = loss_fun(net(u_initial), u_initial_y)
        # computing the loss of u(x,0)

        loss2 = loss_fun(net(u_boundary_left), u_boundary_left_y)
        # computing the loss of u(0,t)

        loss3 = loss_fun(net(u_boundary_right), u_boundary_right_y)
        # computing the loss of u(L,t)

        loss4 = loss_fun(LHS, RHS)
        # computing the loss of u_t + 2u_x = 0

        loss = loss1 + loss2 + loss3 + loss4
        pde_loss = loss4
        boundary_loss = loss2 + loss3
        initial_loss = loss1


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss.append(loss.item())
        loss_on_pde.append(pde_loss.item())
        loss_on_boundary.append(boundary_loss.item())
        loss_on_initial.append(initial_loss.item())

        if validation_set is not None:
            total_loss_val = []


            loss_val = loss_fun(net(collocation_points_val), collocation_points_y)


            total_loss_val.append(loss_val.item)
            if total_loss_val < min_val_error:
                min_val_error = total_loss_val
                best_epoch = i
                best_model = copy.deepcopy(net)
                best_pde_loss = loss_on_pde
                best_boundary_loss = loss_on_boundary
                best_initial_loss = loss_on_initial
                best_total_loss = total_loss


            if i % 100 == 0:
                print(
                    f'total loss = {loss.item()}     pde_loss = {pde_loss.item()}      boundary_loss = {boundary_loss.item()}      initial_loss = {initial_loss.item()}          val_loss = {loss_val.item()}')

        else:

            if i % 100 == 0:
                print(
                    f'total loss = {loss.item()}     pde_loss = {pde_loss.item()}      boundary_loss = {boundary_loss.item()}      initial_loss = {initial_loss.item()}')

    if save == True:
        if not os.path.isdir(path):
            model_path = os.path.join(path, 'data/' + model_name)
            os.makedirs(model_path, exist_ok = True)
            os.makedirs(model_path + '/Loss', exist_ok=True)
            os.makedirs(model_path + '/training_data', exist_ok=True)

        torch.save(net, model_path + f'/{model_name}.pt')

        loss_path = model_path + '/Loss'
        save_data(total_loss,'total_loss',loss_path)
        save_data(loss_on_pde, 'pde_loss', loss_path)
        save_data(loss_on_initial, 'initial_loss', loss_path)
        save_data(loss_on_boundary, 'boundary_loss', loss_path)

        training_path = model_path + '/training_data'
        save_data(u_initial_y,'intial_condition_y',training_path)
        save_data(u_boundary_left_y,'boundary_left_y',training_path)
        save_data(u_boundary_right_y,'boundary_right_y',training_path)
        save_data(collocation_points_y,'collocation_points_y',training_path)
        #save_data(pde,'pde_object',training_path)

        if validation_set is not None:
            best_model_path = model_path + '/model_best'
            torch.save(best_model, best_model_path + f'/{model_name}.pt')
            save_data(best_epoch, 'best_epoch', loss_path)
            save_data(total_loss_val,'loss_val',loss_path)





