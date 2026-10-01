import copy
import torch

def average_weight(model_set, sample_number, server_model):
    weight_average = copy.deepcopy(model_set[0].state_dict())
    for key in weight_average.keys():
        weight_average[key] = model_set[0].state_dict()[key] * sample_number[0]
        for i in range(1, len(sample_number)):
            weight_average[key] += model_set[i].state_dict()[key] * sample_number[i]
        weight_average[key] = torch.div(weight_average[key], sum(sample_number))
    
    server_model.load_state_dict(weight_average)
    return server_model

def average_weight_scaffold(model_set, sample_number, server_model, args):
    weight_average = copy.deepcopy(model_set[0].state_dict())
    x = copy.deepcopy(model_set[0].delta_y)
    c = copy.deepcopy(model_set[0].delta_control)
    for key in weight_average.keys():
        weight_average[key] = model_set[0].state_dict()[key] * sample_number[0]
        x[key] = model_set[0].delta_y[key] * sample_number[0]
        c[key] = model_set[0].delta_control[key] * sample_number[0]
        for i in range(1, len(sample_number)):
            weight_average[key] += model_set[i].state_dict()[key] * sample_number[i]
            x[key] += model_set[i].delta_y[key] * sample_number[i]
            c[key] += model_set[i].delta_control[key] * sample_number[i]
        weight_average[key] = torch.div(weight_average[key], sum(sample_number))
        x[key] = torch.div(x[key], sum(sample_number))
        c[key] = torch.div(c[key], sum(sample_number))
    
    server_model.load_state_dict(weight_average)
    
    for k, v in server_model.named_parameters():
        v.data = v.data + x[k].data
        server_model.control[k].data = server_model.control[k].data + c[k].data * (len(model_set) / (args.terminal_number))
        server_model.delta_control[k].data = c[k].data
        server_model.delta_y[k].data = x[k].data

    return server_model

def average_weight_fednova(model_set, sample_number, server_model):
    server_model_weight = copy.deepcopy(server_model.state_dict())
    coefficient = 0.0

    for i in range(len(sample_number)):
        coefficient += model_set[i].coefficient * sample_number[i] / sum(sample_number)

    server_normal_gradient = copy.deepcopy(model_set[0].normal_gradient)
    for key in server_normal_gradient.keys():
        server_normal_gradient[key] = model_set[0].normal_gradient[key] * sample_number[0]
        for i in range(1, len(sample_number)):
            server_normal_gradient[key] += model_set[i].normal_gradient[key] * sample_number[i]
        server_normal_gradient[key] = torch.div(server_normal_gradient[key], sum(sample_number))
    
    for key in server_model_weight:
        server_model_weight[key] -= coefficient * server_normal_gradient[key]
        server_model.normal_gradient[key] = server_normal_gradient[key]
    
    server_model.coefficient = coefficient
    server_model.load_state_dict(server_model_weight)

    return server_model

def exp_details(args):
    print('\nSimulation details:')
    print(f'    Model     : MLP')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.learning_rate}')
    print(f'    Global Rounds   : {args.global_epoch}\n')

    print('    Hierarchical federated learning parameters:')
    print('    Non-IID')
    print(f'    Fraction of users  : {args.user_fraction}')
    print(f'    Local Batch size   : {args.local_batch_size}')
    print(f'    Local Epochs       : {args.local_epoch}')
    print(f'    Mid Epochs       : {args.mid_epoch}\n')
    return