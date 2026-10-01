import torch
from torch import nn
import torch.optim
from torch.utils.data import DataLoader
from dataset import Dataset_Split
import copy
from scaffold_optimizer import Scaffold_Optimizer

class Local_Update():
    def __init__(self, args, dataset, index, logger):
        self.args = args
        self.logger = logger
        self.train_loader, self.test_loader = self.train_test(dataset, list(index))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def train_test(self, dataset, index):
        train_index = index[:int(0.8 * len(index))]
        test_index = index[int(0.8 * len(index)):]

        train_loader = DataLoader(Dataset_Split(dataset, train_index), batch_size = self.args.local_batch_size, shuffle = False)
        test_loader = DataLoader(Dataset_Split(dataset, test_index), batch_size = max(int(len(test_index) / 10), 1), shuffle = False)
        return train_loader, test_loader
    
    def update_weight(self, model):
        model.train()
        epoch_loss = []

        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = self.args.learning_rate)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.args.learning_rate)

        for _ in range(self.args.local_epoch):
            batch_loss = []
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                prediction = model(data)
                loss = self.criterion(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss = 0.0
        total = 0.0
        correct = 0.0

        for data, label in self.test_loader:
            data, label = data.to(self.device), label.to(self.device)

            output = model(data)
            batch_loss = self.criterion(output, label)
            loss += batch_loss.item()

            _, prediction_label = torch.max(output, 1)
            prediction_label = prediction_label.view(-1)
            correct += torch.sum(torch.eq(prediction_label, label)).item()
            total += len(label)
        
        accuracy = correct / total
        return accuracy, loss
    
class Local_Update_FedProx(Local_Update):
    def __init__(self, args, dataset, index, logger):
        super().__init__(args, dataset, index, logger)

    def update_weight(self, user_model, intermediate_server_model):
        user_model.train()
        global_model = copy.deepcopy(intermediate_server_model)
        epoch_loss = []
        
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(user_model.parameters(), lr = self.args.learning_rate)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(user_model.parameters(), lr = self.args.learning_rate)
    
        for _ in range(self.args.local_epoch):
            batch_loss = []
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                proximal_term = 0.0
                for weight, weight_t in zip(user_model.parameters(), global_model.parameters()):
                    proximal_term += (weight - weight_t).norm(2)
                
                prediction = user_model(data)
                loss = self.criterion(prediction, label) + (self.args.mu_moon / 2) * proximal_term
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return user_model, sum(epoch_loss) / len(epoch_loss)
    
class Local_Update_Scaffold(Local_Update):
    def __init__(self, args, dataset, index, logger):
        super().__init__(args, dataset, index, logger)
        self.data_total = len(dataset) * 0.01 * 0.8

    def update_weight(self, user_model, intermediate_server_model):
        user_model.train()
        epoch_loss = []
        x = copy.deepcopy(user_model)

        '''
        for k in user_model.control:
            if k == 'layer3.bias':
                print('user_model', user_model.control[k])
                print('intermediate_server_model', intermediate_server_model.control[k])
        '''

        c_difference = {key: user_model.control[key] - intermediate_server_model.control[key] for key in user_model.control.keys()}

        if self.args.optimizer == 'scaffold':
            optimizer = Scaffold_Optimizer(user_model.parameters(), self.args.learning_rate)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(user_model.parameters(), lr = self.args.learning_rate)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(user_model.parameters(), lr = self.args.learning_rate)
        
        for _ in range(self.args.local_epoch):
            batch_loss = []
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                prediction = user_model(data)
                loss = self.criterion(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                if self.args.optimizer == 'scaffold':
                    optimizer.step(intermediate_server_model.control, user_model.control)
                else:
                    for (key, value) in user_model.named_parameters():
                        if value.grad is not None:
                            value.grad -= c_difference[key]
                    optimizer.step()
            
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        temporary = {}
        for k, v in user_model.named_parameters():
            temporary[k] = v.data.clone()
        
        for k, v in x.named_parameters():
            user_model.control[k] = user_model.control[k] - intermediate_server_model.control[k] + (v.data - temporary[k]) / (50 * self.data_total * self.args.local_epoch * self.args.learning_rate)
            user_model.delta_y[k] = temporary[k] - v.data
            user_model.delta_control[k] = user_model.control[k] - x.control[k]

        return user_model, sum(epoch_loss) / len(epoch_loss)

class Local_Update_FedNova(Local_Update):
    def __init__(self, args, dataset, index, logger):
        super().__init__(args, dataset, index, logger)

    def update_weight(self, user_model, intermediate_server_model):
        user_model.train()
        epoch_loss = []
        intermediate_server_model_weight = copy.deepcopy(intermediate_server_model.state_dict())

        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(user_model.parameters(), lr = self.args.learning_rate)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(user_model.parameters(), lr = self.args.learning_rate)

        tau = 0
        for _ in range(self.args.local_epoch):
            batch_loss = []
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                prediction = user_model(data)
                loss = self.criterion(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tau += 1
            
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
    
        coefficient = (tau - self.args.rho * (1 - pow(self.args.rho, tau)) / (1 - self.args.rho)) / (1 - self.args.rho)
        user_model.coefficient = coefficient

        user_model_weight = copy.deepcopy(user_model.state_dict())
        normal_gradient = copy.deepcopy(intermediate_server_model_weight)
        for key in normal_gradient:
            normal_gradient[key] = torch.div(intermediate_server_model_weight[key] - user_model_weight[key], coefficient)
        
        for key in normal_gradient:
            user_model.normal_gradient[key] = normal_gradient[key]

        return user_model, sum(epoch_loss) / len(epoch_loss)

class Local_Update_MOON(Local_Update):
    def __init__(self, args, dataset, index, logger):
        super().__init__(args, dataset, index, logger)
    
    def update_weight(self, user_model, intermediate_server_model, previous_local_model_set):
        user_model.train()
        epoch_loss = []

        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(user_model.parameters(), lr = self.args.learning_rate)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(user_model.parameters(), lr = self.args.learning_rate)
        
        cosine_similarity = nn.CosineSimilarity(dim = -1)
        
        for _ in range(self.args.local_epoch):
            batch_loss = []
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                _, projection_1, prediction = user_model(data)
                _, projection_2, _ = intermediate_server_model(data)

                positive_distance = cosine_similarity(projection_1, projection_2)
                logits = positive_distance.reshape(-1, 1)

                for previous_local_model in previous_local_model_set:
                    previous_local_model.to(self.device)
                    _, projection_3, _ = previous_local_model(data)
                    negative_distance = cosine_similarity(projection_1, projection_3)
                    logits = torch.cat((logits, negative_distance.reshape(-1, 1)), dim = 1)

                    previous_local_model.to(self.device)
                
                logits /= self.args.t_moon
                labels = torch.zeros(data.size(0)).to(self.device).long()

                loss_2 = self.args.mu_moon * self.criterion(logits, labels)

                loss_1 = self.criterion(prediction, label)
                loss = loss_1 + loss_2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return user_model, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss = 0.0
        total = 0.0
        correct = 0.0

        for data, label in self.test_loader:
            data, label = data.to(self.device), label.to(self.device)

            _, _, output = model(data)
            batch_loss = self.criterion(output, label)
            loss += batch_loss.item()

            _, prediction_label = torch.max(output, 1)
            prediction_label = prediction_label.view(-1)
            correct += torch.sum(torch.eq(prediction_label, label)).item()
            total += len(label)
        
        accuracy = correct / total
        return accuracy, loss

def test_inference(args, model, test_dataset):
    model.eval()
    loss = 0.0
    total = 0.0
    correct = 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    test_loader = DataLoader(test_dataset, args.local_batch_size, shuffle = False)

    prediction_test = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        output = model(data)
        batch_loss = criterion(output, label)
        loss += batch_loss.item()

        _, prediction_label = torch.max(output, 1)
        prediction_label = prediction_label.view(-1)
        correct += torch.sum(torch.eq(prediction_label, label)).item()
        total += len(label)
        for index in range(len(prediction_label)):
            prediction_test.append(int(prediction_label[index]))

    accuracy = correct / total
    return accuracy, loss, prediction_test

def test_inference_moon(args, model, test_dataset):
    model.eval()
    loss = 0.0
    total = 0.0
    correct = 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    test_loader = DataLoader(test_dataset, args.local_batch_size, shuffle = False)

    prediction_test = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        _, _, output = model(data)
        batch_loss = criterion(output, label)
        loss += batch_loss.item()

        _, prediction_label = torch.max(output, 1)
        prediction_label = prediction_label.view(-1)
        correct += torch.sum(torch.eq(prediction_label, label)).item()
        total += len(label)
        for index in range(len(prediction_label)):
            prediction_test.append(int(prediction_label[index]))

    accuracy = correct / total
    return accuracy, loss, prediction_test