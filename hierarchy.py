import numpy as np
import copy
from utils import average_weight
from utils import average_weight_scaffold
from utils import average_weight_fednova
from option import args_parser
import math
import random
from dataset import Application_Information_Normal
from dataset import Application_Information_Propagation_Loss
from torch.utils.data import DataLoader
import torch

class Structure(object):
    def __init__(self, args, global_model, test_dataset = None):
        self.nn_architecture = global_model
        self.args = args
        self.dataset = test_dataset
        self.system = self.hierarchy()
        self.layer = len(args.mid_server) + 1
    
    def hierarchy(self):
        """
        the hierarchical structure is represented as follows
        { -1: {user_idx: local weight} # user layer
          0: {server_idx: [user_set, aggregated weight]} # edge server layer
          1: {sever_idx: [server_set, aggregated weight]} # higher server layer
          ...
        }
        """
        structure = {}

        def build(layer_index):

            layer_dictionary = {}

            if layer_index == -1:
                for user in range(self.args.terminal_number):
                    layer_dictionary[user] = self.nn_architecture
                
            if layer_index == len(self.args.mid_server):
                server_set = [i for i in range(self.args.mid_server[layer_index - 1])]
                layer_dictionary[0] = [server_set, self.nn_architecture]
                structure[layer_index] = layer_dictionary
                return

            server_number = self.args.mid_server[layer_index]

            if layer_index == 0:
                all_user = [i for i in range(self.args.terminal_number)]
                user_per_server = self.args.terminal_number // server_number

                for server in range(server_number):
                    if server == (server_number - 1):
                        user_set = set(all_user)
                    else:
                        user_set = set(np.random.choice(all_user, user_per_server, replace = False))
                    layer_dictionary[server] = [user_set, self.nn_architecture]
                    all_user = list(set(all_user) - user_set)
            
            elif layer_index > 0:
                previous_server_number = self.args.mid_server[layer_index - 1]
                server_per_server = previous_server_number // server_number
                all_server = [i for i in range(previous_server_number)]

                for server in range(server_number):
                    if server == (server_number - 1):
                        server_set = set(all_server)
                    else:
                        server_set = set(np.random.choice(all_server, server_per_server, replace = False))

                    layer_dictionary[server] = [server_set, self.nn_architecture]
                    all_server = list(set(all_server) - server_set)

            structure[layer_index] = layer_dictionary
            build(layer_index + 1)
        
        build(-1)

        return structure
    
    def get_model(self, user_index, download):
        intermediate_server = self.system[0]

        for intermediate_server_index in intermediate_server.keys():
            if user_index in intermediate_server[intermediate_server_index][0]:
                if download:
                    intermediate_server_model = copy.deepcopy(intermediate_server[intermediate_server_index][1])
                else:
                    intermediate_server_model = copy.deepcopy(self.system[-1][user_index])
                
                return intermediate_server_model, intermediate_server_index

    def upload_weight(self, local_weight, user_server_index, user_group, mid_epoch):
        def update(layer_index):
            if layer_index == self.layer:
                return

            if layer_index == 0:
                for server_index in local_weight.keys():
                    sample_number = []
                    for user_index in user_server_index[server_index]:
                        sample_number.append(int(0.8 * len(user_group[user_index])))
                    intermediate_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                    self.system[layer_index][server_index][1] = average_weight(local_weight[server_index], sample_number, intermediate_server_model)

            else:
                if mid_epoch != self.args.mid_epoch - 1:
                    return

                else:
                    for server_index in self.system[layer_index]:
                        sample_number = []
                        previous_server_model = []
                        for previous_server_index in user_server_index:
                            count_sample = 0
                            for user_index in user_server_index[previous_server_index]:
                                count_sample += int(0.8 * len(user_group[user_index]))
                            sample_number.append(count_sample)
                            previous_server_model.append(self.system[layer_index - 1][previous_server_index][1])

                        centralized_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                        self.system[layer_index][server_index][1] = average_weight(previous_server_model, sample_number, centralized_server_model)

                        for previous_server_index in self.system[layer_index][server_index][0]:
                            self.system[layer_index - 1][previous_server_index][1] = self.system[layer_index][server_index][1]

            update(layer_index + 1)

        update(0)

class Structure_Scaffold(Structure):
    def __init__(self, args, global_model, test_dataset = None):
        super().__init__(args, global_model, test_dataset)
        self.system = self.hierarchy()
        self.layer = len(args.mid_server) + 1
        
    def get_model(self, user_index, download):
        intermediate_server = self.system[0]

        for intermediate_server_index in intermediate_server.keys():
            if user_index in intermediate_server[intermediate_server_index][0]:
                if download:
                    for old_parameter, new_parameter in zip(self.system[-1][user_index].parameters(), intermediate_server[intermediate_server_index][1].parameters()):
                        old_parameter.data = new_parameter.data.clone()
                    user_model = copy.deepcopy(self.system[-1][user_index])
                    intermediate_server_model = copy.deepcopy(intermediate_server[intermediate_server_index][1])
                    
                    return user_model, intermediate_server_model, intermediate_server_index

                else:
                    intermediate_server_model = copy.deepcopy(self.system[-1][user_index])
                
                    return user_model, intermediate_server_model, intermediate_server_index
            
    def upload_weight(self, local_weight, user_server_index, user_group, mid_epoch):
        def update(layer_index):
            if layer_index == self.layer:
                return
            
            if layer_index == 0:
                for server_index in local_weight.keys():
                    sample_number = []
                    for user_index in user_server_index[server_index]:
                        sample_number.append(int(0.8 * len(user_group[user_index])))
                    intermediate_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                    self.system[layer_index][server_index][1] = average_weight_scaffold(local_weight[server_index], sample_number, intermediate_server_model, self.args)

            else:
                if mid_epoch != self.args.mid_epoch - 1:
                    return
                
                else:
                    for server_index in self.system[layer_index]:
                        sample_number = []
                        previous_server_model = []
                        for previous_server_index in user_server_index:
                            count_sample = 0
                            for user_index in user_server_index[previous_server_index]:
                                count_sample += int(0.8 * len(user_group[user_index]))
                            sample_number.append(count_sample)
                            previous_server_model.append(self.system[layer_index - 1][previous_server_index][1])

                        centralized_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                        self.system[layer_index][server_index][1] = average_weight_scaffold(previous_server_model, sample_number, centralized_server_model, self.args)
                        
                        for previous_server_index in self.system[layer_index][server_index][0]:
                            self.system[layer_index - 1][previous_server_index][1] = copy.deepcopy(self.system[layer_index][server_index][1])
                        
            update(layer_index + 1)

        update(0)

class Structure_FedNova(Structure):
    def __init__(self, args, global_model, test_dataset = None):
        super().__init__(args, global_model, test_dataset)
        self.system = self.hierarchy()
        self.layer = len(args.mid_server) + 1
            
    def upload_weight(self, local_weight, user_server_index, user_group, mid_epoch):
        def update(layer_index):
            if layer_index == self.layer:
                return
            
            if layer_index == 0:
                for server_index in local_weight.keys():
                    sample_number = []
                    for user_index in user_server_index[server_index]:
                        sample_number.append(int(0.8 * len(user_group[user_index])))
                    intermediate_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                    self.system[layer_index][server_index][1] = average_weight_fednova(local_weight[server_index], sample_number, intermediate_server_model)

            else:
                if mid_epoch != self.args.mid_epoch - 1:
                    return
                
                else:
                    for server_index in self.system[layer_index]:
                        sample_number = []
                        previous_server_model = []
                        for previous_server_index in user_server_index:
                            count_sample = 0
                            for user_index in user_server_index[previous_server_index]:
                                count_sample += int(0.8 * len(user_group[user_index]))
                            sample_number.append(count_sample)
                            previous_server_model.append(self.system[layer_index - 1][previous_server_index][1])
                        
                        # print(previous_server_model[0].normal_gradient)

                        centralized_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                        self.system[layer_index][server_index][1] = average_weight_fednova(previous_server_model, sample_number, centralized_server_model)
                        
                        for previous_server_index in self.system[layer_index][server_index][0]:
                            self.system[layer_index - 1][previous_server_index][1] = self.system[layer_index][server_index][1]
                        
            update(layer_index + 1)

        update(0)

class Structure_FedDF(Structure):
    def __init__(self, args, global_model, test_dataset = None):
        super().__init__(args, global_model, test_dataset)
        self.system = self.hierarchy()
        self.layer = len(args.mid_server) + 1
            
    def upload_weight(self, local_weight, user_server_index, user_group, mid_epoch, fitted_kpca = None):
        def update(layer_index):
            if layer_index == self.layer:
                return
            
            if layer_index == 0:
                for server_index in local_weight.keys():
                    sample_number = []
                    for user_index in user_server_index[server_index]:
                        sample_number.append(int(0.8 * len(user_group[user_index])))
                    intermediate_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                    self.system[layer_index][server_index][1] = average_weight(local_weight[server_index], sample_number, intermediate_server_model)
                
                self.distillation_intermediate_server(layer_index, local_weight, fitted_kpca)

            else:
                if mid_epoch != self.args.mid_epoch - 1:
                    return
                
                else:
                    for server_index in self.system[layer_index]:
                        sample_number = []
                        previous_server_model = []
                        for previous_server_index in user_server_index:
                            count_sample = 0
                            for user_index in user_server_index[previous_server_index]:
                                count_sample += int(0.8 * len(user_group[user_index]))
                            sample_number.append(count_sample)
                            previous_server_model.append(self.system[layer_index - 1][previous_server_index][1])

                        centralized_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                        self.system[layer_index][server_index][1] = average_weight(previous_server_model, sample_number, centralized_server_model)
                        
                    self.distillation_centralized_server(layer_index, user_server_index, fitted_kpca)
                        
            update(layer_index + 1)

        update(0)
    
    def distillation_intermediate_server(self, layer_index, local_weight, fitted_kpca):
        device = 'cuda' if self.args.gpu else 'cpu'

        for server_index in local_weight.keys():
            application_id_random_set = []
            for _ in range(self.args.unlabel_data_number):
                application_id_random = math.floor(random.random() * self.args.application_number)
                application_id_random_set.append(application_id_random)
            
            if self.args.propagation_loss == 'normal':
                unlabel_data = Application_Information_Normal(self.args, application_id_random_set)
                unlabel_data_set = unlabel_data.application_information_random_data()
            elif self.args.propagation_loss == 'fspl':
                unlabel_data = Application_Information_Propagation_Loss(self.args, application_id_random_set, fitted_kpca)
                unlabel_data_set = unlabel_data.application_information_random_data()
            elif self.args.propagation_loss == 'nfspl':
                unlabel_data = Application_Information_Propagation_Loss(self.args, application_id_random_set, fitted_kpca)
                unlabel_data_set = unlabel_data.application_information_random_data()

            unlabel_data_set_loader = DataLoader(unlabel_data_set, self.args.unlabel_data_number, shuffle = False)

            intermediate_server_model = copy.deepcopy(self.system[layer_index][server_index][1])

            intermediate_server_model.to(device)
            intermediate_server_model.train()

            if self.args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(intermediate_server_model.parameters(), lr = self.args.learning_rate)
            elif self.args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(intermediate_server_model.parameters(), lr = self.args.learning_rate)

            loss_function = torch.nn.KLDivLoss(reduction='batchmean').to(device)

            for unlabel_data in unlabel_data_set_loader:
                teacher_logits_set = []
                unlabel_data = unlabel_data.to(device)
                for user_model in local_weight[server_index]:
                    user_model.eval()
                    with torch.no_grad():
                        teacher_logits = user_model(unlabel_data)
                        teacher_logits_set.append(teacher_logits)
                    
                teacher_logits_average = torch.mean(torch.stack(teacher_logits_set), dim=0)
                student_logits = intermediate_server_model(unlabel_data)

                temperature = self.args.t_feddf
                loss = loss_function(
                    torch.nn.functional.log_softmax(student_logits / temperature, dim=1),
                    torch.nn.functional.softmax(teacher_logits_average / temperature, dim=1),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.system[layer_index][server_index][1] = intermediate_server_model
    
    def distillation_centralized_server(self, layer_index, user_server_index, fitted_kpca):
        device = 'cuda' if self.args.gpu else 'cpu'

        application_id_random_set = []
        for _ in range(self.args.unlabel_data_number):
            application_id_random = math.floor(random.random() * self.args.application_number)
            application_id_random_set.append(application_id_random)
            
        if self.args.propagation_loss == 'normal':
            unlabel_data = Application_Information_Normal(self.args, application_id_random_set)
            unlabel_data_set = unlabel_data.application_information_random_data()
        elif self.args.propagation_loss == 'fspl':
            unlabel_data = Application_Information_Propagation_Loss(self.args, application_id_random_set, fitted_kpca)
            unlabel_data_set = unlabel_data.application_information_random_data()
        elif self.args.propagation_loss == 'nfspl':
            unlabel_data = Application_Information_Propagation_Loss(self.args, application_id_random_set, fitted_kpca)
            unlabel_data_set = unlabel_data.application_information_random_data()

        unlabel_data_set_loader = DataLoader(unlabel_data_set, self.args.unlabel_data_number, shuffle = False)

        for server_index in self.system[layer_index]:
            previous_server_model_set = []
            for previous_server_index in user_server_index:
                previous_server_model_set.append(self.system[layer_index - 1][previous_server_index][1])

            centralized_server_model = copy.deepcopy(self.system[layer_index][server_index][1])

            centralized_server_model.to(device)
            centralized_server_model.train()

            if self.args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(centralized_server_model.parameters(), lr = self.args.learning_rate)
            elif self.args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(centralized_server_model.parameters(), lr = self.args.learning_rate)
            
            loss_function = torch.nn.KLDivLoss(reduction='batchmean').to(device)
            
            for unlabel_data in unlabel_data_set_loader:
                teacher_logits_set = []
                unlabel_data = unlabel_data.to(device)
                for previous_server_model in previous_server_model_set:
                    previous_server_model.eval()
                    with torch.no_grad():
                        teacher_logits = previous_server_model(unlabel_data)
                        teacher_logits_set.append(teacher_logits)
                
                teacher_logits_average = torch.mean(torch.stack(teacher_logits_set), dim = 0)
                student_logits = centralized_server_model(unlabel_data)

                # print(unlabel_data)
                # print(len(unlabel_data))

                temperature = self.args.t_feddf
                loss = loss_function(
                    torch.nn.functional.log_softmax(student_logits / temperature, dim = 1),
                    torch.nn.functional.softmax(teacher_logits_average / temperature, dim = 1),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.system[layer_index][server_index][1] = centralized_server_model

        for server_index in self.system[layer_index]:
            for previous_server_index in self.system[layer_index][server_index][0]:
                self.system[layer_index - 1][previous_server_index][1] = self.system[layer_index][server_index][1]
        
        # for k, v in self.system[0][0][1].named_parameters():
            # if k == 'layer3.bias':
                # print(v)

class Structure_MOON(Structure):
    def __init__(self, args, global_model, test_dataset = None):
        super().__init__(args, global_model, test_dataset)
        self.system = self.hierarchy()
        self.layer = len(args.mid_server) + 1
        self.previous_local_model_set = []

    def get_model(self, user_index, download):
        previous_local_model = []
        for id in range(len(self.previous_local_model_set)):
            previous_local_model.append(self.previous_local_model_set[id][user_index])

        intermediate_server = self.system[0]
        for intermediate_server_index in intermediate_server.keys():
            if user_index in intermediate_server[intermediate_server_index][0]:
                if download:
                    intermediate_server_model = copy.deepcopy(intermediate_server[intermediate_server_index][1])
                else:
                    intermediate_server_model = copy.deepcopy(self.system[-1][user_index])
                
                return previous_local_model, intermediate_server_model, intermediate_server_index

    def upload_weight(self, local_weight, user_server_index, user_group, mid_epoch):
        def update(layer_index):
            if layer_index == self.layer:
                return
            
            if layer_index == 0:
                for server_index in local_weight.keys():
                    sample_number = []
                    for user_index in user_server_index[server_index]:
                        sample_number.append(int(0.8 * len(user_group[user_index])))
                    intermediate_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                    self.system[layer_index][server_index][1] = average_weight(local_weight[server_index], sample_number, intermediate_server_model)

                if len(self.previous_local_model_set) < self.args.model_buffer_size:
                    local_model_set_all = copy.deepcopy(self.system[-1])
                    for _, local_model in local_model_set_all.items():
                        local_model.eval()
                        for param in local_model.parameters():
                            param.requires_grad = False
                    self.previous_local_model_set.append(local_model_set_all)
                elif self.args.previous_model_pool_option == 'FIFO':
                    local_model_set_all = copy.deepcopy(self.system[-1])
                    for _, local_model in local_model_set_all.items():
                        local_model.eval()
                        for param in local_model.parameters():
                            param.requires_grad = False
                    for i in range(self.args.model_buffer_size -2, -1, -1):
                        self.previous_local_model_set[i] = self.previous_local_model_set[i + 1]
                    self.previous_local_model_set[self.args.model_buffer_size - 1] = local_model_set_all

            else:
                if mid_epoch != self.args.mid_epoch - 1:
                    return

                else:
                    for server_index in self.system[layer_index]:
                        sample_number = []
                        previous_server_model = []
                        for previous_server_index in user_server_index:
                            count_sample = 0
                            for user_index in user_server_index[previous_server_index]:
                                count_sample += int(0.8 * len(user_group[user_index]))
                            sample_number.append(count_sample)
                            previous_server_model.append(self.system[layer_index - 1][previous_server_index][1])

                        centralized_server_model = copy.deepcopy(self.system[layer_index][server_index][1])
                        self.system[layer_index][server_index][1] = average_weight(previous_server_model, sample_number, centralized_server_model)

                        for previous_server_index in self.system[layer_index][server_index][0]:
                            self.system[layer_index - 1][previous_server_index][1] = self.system[layer_index][server_index][1]

            update(layer_index + 1)

        update(0)

if __name__ == '__main__':
    args = args_parser()
    structure = Structure(args, 'a', 2)
    print(structure.system, "\n")