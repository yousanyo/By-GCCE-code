from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, input_number, hidden_number, output_number):
        super().__init__()
        self.layer1 = nn.Linear(input_number, hidden_number)
        self.layer2 = nn.Linear(hidden_number, hidden_number)
        self.layer3 = nn.Linear(hidden_number, output_number)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

class MLP_Scaffold(MLP):
    def __init__(self, input_number, hidden_number, output_number):
        super().__init__(input_number, hidden_number, output_number)
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

class MLP_FedNova(MLP):
    def __init__(self, input_number, hidden_number, output_number):
        super().__init__(input_number, hidden_number, output_number)
        self.coefficient = 0.0
        self.normal_gradient = {}

class MLP_Moon(nn.Module):
    def __init__(self, input_number, hidden_number, output_number, projection_number = 128):
        super().__init__()
        self.layer1 = nn.Linear(input_number, hidden_number)
        self.layer2 = nn.Linear(hidden_number, hidden_number)
        self.layer3 = nn.Linear(projection_number, output_number)
        self.relu = nn.ReLU()

        self.projection1 = nn.Linear(hidden_number, hidden_number)
        self.projection2 = nn.Linear(hidden_number, projection_number)
    
    def forward(self, x):
        h = self.relu(self.layer2(self.relu(self.layer1(x))))
        x = self.projection2(self.relu(self.projection1(h)))
        y = self.layer3(x)
        return h, x, y

def mlp(args, global_model, feature_set):
    device = 'cuda' if args.gpu else 'cpu'
    global_model.eval()

    feature_set = feature_set.to(device)
    output = global_model(feature_set)
    _, prediction = torch.max(output, 1)
    prediction = prediction.view(-1)

    # print(f'Access point id after allocation: {prediction}')

    return prediction

def mlp_moon(args, global_model, feature_set):
    device = 'cuda' if args.gpu else 'cpu'
    global_model.eval()

    feature_set = feature_set.to(device)
    _, _, output = global_model(feature_set)
    _, prediction = torch.max(output, 1)
    prediction = prediction.view(-1)

    # print(f'Access point id after allocation: {prediction}')

    return prediction