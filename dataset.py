import json
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import random
import math
from sklearn.decomposition import KernelPCA

with open('application.json', 'r') as f:
    application_type = json.load(f)

class Feature_Random_Normal():
    def __init__(self, args, terminal_set):
        self.args = args
        self.terminal_set = terminal_set

    def application_information_random_data(self):
        tp_need = []
        rtt_need = []
        application_id = []

        for terminal in self.terminal_set:
            tp_need.append(terminal.tp_need)
            rtt_need.append(terminal.rtt_need)
            application_id.append(terminal.application_id)
        
        tp_need = np.array(tp_need)
        rtt_need = np.array(rtt_need)
        tp_need_norm = self.standardization(tp_need)
        rtt_need_norm = self.standardization(rtt_need)
        network_indicator = torch.stack([tp_need_norm, rtt_need_norm], axis = 1)

        application_number = self.args.application_number
        application_id = torch.tensor(application_id)
        application_id_encoded = one_hot(application_id % application_number)
        feature_set = torch.cat([network_indicator, application_id_encoded], 1).float()

        return feature_set

    def standardization(self, data):
        train_data_std = (data - np.mean(data)) / np.std(data)
        return torch.from_numpy(train_data_std).float()

class Feature_Random_Propagation_Loss(Feature_Random_Normal):
    def __init__(self, args, terminal_set, rssi_all, fitted_kpca=None):
        super().__init__(args, terminal_set)
        self.rssi_all = rssi_all
        self.fitted_kpca = fitted_kpca

    def application_information_random_data(self):
        tp_need = []
        rtt_need = []
        application_id = []
        rssi_access_point_0 = []
        rssi_access_point_1 = []
        rssi_access_point_2 = []

        for index, terminal in enumerate(self.terminal_set):
            tp_need.append(terminal.tp_need)
            rtt_need.append(terminal.rtt_need)
            application_id.append(terminal.application_id)
            rssi_access_point_0.append(self.rssi_all[index][0])
            rssi_access_point_1.append(self.rssi_all[index][1])
            rssi_access_point_2.append(self.rssi_all[index][2])
        
        tp_need = np.array(tp_need)
        rtt_need = np.array(rtt_need)
        tp_need_norm = self.standardization(tp_need)
        rtt_need_norm = self.standardization(rtt_need)
        network_indicator = torch.stack([tp_need_norm, rtt_need_norm], axis = 1)

        application_number = self.args.application_number
        application_id = torch.tensor(application_id)
        application_id_encoded = one_hot(application_id % application_number)
        application_information = torch.cat([network_indicator, application_id_encoded], 1).float()

        rssi_access_point_0 = np.array(rssi_access_point_0)
        rssi_access_point_1 = np.array(rssi_access_point_1)
        rssi_access_point_2 = np.array(rssi_access_point_2)
        rssi_access_point_0_norm = self.standardization(rssi_access_point_0)
        rssi_access_point_1_norm = self.standardization(rssi_access_point_1)
        rssi_access_point_2_norm = self.standardization(rssi_access_point_2)

        if self.args.kpca:
            rssi_stacked = torch.stack([rssi_access_point_0_norm, rssi_access_point_1_norm, rssi_access_point_2_norm], axis=1)
            rssi_kpca_np = self.fitted_kpca.transform(rssi_stacked.numpy())

            rssi_kpca = torch.tensor(rssi_kpca_np, dtype=torch.float32)
            feature_set = torch.cat([application_information, rssi_kpca], dim=1)

        else:
            rssi_access_point_0_norm = rssi_access_point_0_norm.unsqueeze(1)
            rssi_access_point_1_norm = rssi_access_point_1_norm.unsqueeze(1)
            rssi_access_point_2_norm = rssi_access_point_2_norm.unsqueeze(1)
            
            stack = torch.cat([application_information, rssi_access_point_0_norm], dim = 1)
            stack = torch.cat([stack, rssi_access_point_1_norm], dim = 1)
            feature_set = torch.cat([stack, rssi_access_point_2_norm], dim = 1)

        return feature_set

class Supervised_Data_Normal:
    def __init__(self, args, file_name):
        self.args = args
        self.file_name = file_name

    def supervised_data(self):
        tp_need, rtt_need, application_number, combination_terminal_ap = self.read_file(self.file_name)

        if self.args.learning_method == 'cl' and self.args.data_number == 4270:
            train_tp_need, test_tp_need, train_rtt_need, test_rtt_need, train_application_number, test_application_number, train_combination_terminal_ap, test_combination_terminal_ap = train_test_split(tp_need, rtt_need, application_number, combination_terminal_ap, test_size = 25./61)
        else:
            train_tp_need, test_tp_need, train_rtt_need, test_rtt_need, train_application_number, test_application_number, train_combination_terminal_ap, test_combination_terminal_ap = train_test_split(tp_need, rtt_need, application_number, combination_terminal_ap, test_size = 1./5)

        train_tp_need_norm, test_tp_need_norm = self.standardization_train_test(train_tp_need, test_tp_need)
        train_rtt_need_norm, test_rtt_need_norm = self.standardization_train_test(train_rtt_need, test_rtt_need)

        total_application_number = self.args.application_number
        train_application_number_encoded = one_hot(torch.tensor(train_application_number) % total_application_number)
        test_application_number_encoded = one_hot(torch.tensor(test_application_number) % total_application_number)

        train_stack = torch.stack([train_tp_need_norm, train_rtt_need_norm], axis = 1)
        test_stack = torch.stack([test_tp_need_norm, test_rtt_need_norm], axis = 1)

        train_x = torch.cat([train_stack, train_application_number_encoded], 1).float()
        test_x = torch.cat([test_stack, test_application_number_encoded], 1).float()

        train_y = torch.tensor(train_combination_terminal_ap)
        test_y = torch.tensor(test_combination_terminal_ap)

        train_data = TensorDataset(train_x, train_y)
        test_data = TensorDataset(test_x, test_y)

        return train_data, test_data, test_y, train_application_number

    def read_file(self, file_name):
        tp_need = []
        rtt_need = []
        application_number = []
        combination_terminal_ap = []

        with open(file_name, 'r') as f:
            data = csv.reader(f)
            for i, row in enumerate(data):
                if i == self.args.data_number:
                    break
                else:
                    tp_need.append(int(row[0]))
                    rtt_need.append(int(row[1]))
                    application_number.append(int(row[2]))
                    combination_terminal_ap.append(int(row[3]))

        return np.array(tp_need), np.array(rtt_need), np.array(application_number), np.array(combination_terminal_ap)

    def standardization_train_test(self, train_data, test_data):
        mean_train_data = np.mean(train_data)
        std_train_data = np.std(train_data)
        train_data_std = (train_data - mean_train_data) / std_train_data
        test_data_std = (test_data - mean_train_data) / std_train_data
        return torch.from_numpy(train_data_std).float(), torch.from_numpy(test_data_std).float()

class Supervised_Data_Propagation_Loss(Supervised_Data_Normal):
    def __init__(self, args, file_name):
        super().__init__(args, file_name)
        self.kpca = None
    
    def supervised_data(self):
        tp_need, rtt_need, application_number, rssi_access_point_0, rssi_access_point_1, rssi_access_point_2, combination_terminal_ap = self.read_file(self.file_name)

        if self.args.learning_method == 'cl' and self.args.data_number == 4270:
            train_tp_need, test_tp_need, train_rtt_need, test_rtt_need, train_application_number, test_application_number, train_rssi_access_point_0, test_rssi_access_point_0, train_rssi_access_point_1, test_rssi_access_point_1, train_rssi_access_point_2, test_rssi_access_point_2, train_combination_terminal_ap, test_combination_terminal_ap = train_test_split(tp_need, rtt_need, application_number, rssi_access_point_0, rssi_access_point_1, rssi_access_point_2, combination_terminal_ap, test_size = 25./61)
        else:
            train_tp_need, test_tp_need, train_rtt_need, test_rtt_need, train_application_number, test_application_number, train_rssi_access_point_0, test_rssi_access_point_0, train_rssi_access_point_1, test_rssi_access_point_1, train_rssi_access_point_2, test_rssi_access_point_2, train_combination_terminal_ap, test_combination_terminal_ap = train_test_split(tp_need, rtt_need, application_number, rssi_access_point_0, rssi_access_point_1, rssi_access_point_2, combination_terminal_ap, test_size = 1./5)

        train_tp_need_norm, test_tp_need_norm = self.standardization_train_test(train_tp_need, test_tp_need)
        train_rtt_need_norm, test_rtt_need_norm = self.standardization_train_test(train_rtt_need, test_rtt_need)
        train_stack = torch.stack([train_tp_need_norm, train_rtt_need_norm], axis = 1)
        test_stack = torch.stack([test_tp_need_norm, test_rtt_need_norm], axis = 1)

        total_application_number = self.args.application_number
        train_application_number_encoded = one_hot(torch.tensor(train_application_number) % total_application_number)
        test_application_number_encoded = one_hot(torch.tensor(test_application_number) % total_application_number)
        train_cat = torch.cat([train_stack, train_application_number_encoded], 1).float()
        test_cat = torch.cat([test_stack, test_application_number_encoded], 1).float()

        train_rssi_access_point_0_norm, test_rssi_access_point_0_norm = self.standardization_train_test(train_rssi_access_point_0, test_rssi_access_point_0)
        train_rssi_access_point_1_norm, test_rssi_access_point_1_norm = self.standardization_train_test(train_rssi_access_point_1, test_rssi_access_point_1)
        train_rssi_access_point_2_norm, test_rssi_access_point_2_norm = self.standardization_train_test(train_rssi_access_point_2, test_rssi_access_point_2)

        if self.args.kpca:
            train_rssi_stacked = torch.stack([train_rssi_access_point_0_norm, train_rssi_access_point_1_norm, train_rssi_access_point_2_norm], axis=1)
            test_rssi_stacked = torch.stack([test_rssi_access_point_0_norm, test_rssi_access_point_1_norm, test_rssi_access_point_2_norm], axis=1)

            kpca = KernelPCA(n_components=1, kernel='rbf')
            self.kpca = kpca

            train_rssi_kpca_np = kpca.fit_transform(train_rssi_stacked.numpy())
            test_rssi_kpca_np = kpca.transform(test_rssi_stacked.numpy())

            train_rssi_kpca = torch.tensor(train_rssi_kpca_np, dtype=torch.float32)
            test_rssi_kpca = torch.tensor(test_rssi_kpca_np, dtype=torch.float32)

            train_x = torch.cat([train_cat, train_rssi_kpca], dim=1)
            test_x = torch.cat([test_cat, test_rssi_kpca], dim=1)
        else:
            train_rssi_access_point_0_norm = train_rssi_access_point_0_norm.unsqueeze(1)
            train_rssi_access_point_1_norm = train_rssi_access_point_1_norm.unsqueeze(1)
            train_rssi_access_point_2_norm = train_rssi_access_point_2_norm.unsqueeze(1)
            test_rssi_access_point_0_norm = test_rssi_access_point_0_norm.unsqueeze(1)
            test_rssi_access_point_1_norm = test_rssi_access_point_1_norm.unsqueeze(1)
            test_rssi_access_point_2_norm = test_rssi_access_point_2_norm.unsqueeze(1)
            train_stack = torch.cat([train_cat, train_rssi_access_point_0_norm], dim = 1)
            train_stack = torch.cat([train_stack, train_rssi_access_point_1_norm], dim = 1)
            train_x = torch.cat([train_stack, train_rssi_access_point_2_norm], dim = 1)
            test_stack = torch.cat([test_cat, test_rssi_access_point_0_norm], dim = 1)
            test_stack = torch.cat([test_stack, test_rssi_access_point_1_norm], dim = 1)
            test_x = torch.cat([test_stack, test_rssi_access_point_2_norm], dim = 1)
        
        train_y = torch.tensor(train_combination_terminal_ap)
        test_y = torch.tensor(test_combination_terminal_ap)

        train_data = TensorDataset(train_x, train_y)
        test_data = TensorDataset(test_x, test_y)

        return train_data, test_data, test_y, train_application_number

    def read_file(self, file_name):
        tp_need = []
        rtt_need = []
        application_number = []
        rssi_access_point_0 = []
        rssi_access_point_1 = []
        rssi_access_point_2 = []
        combination_terminal_ap = []

        with open(file_name, 'r') as f:
            data = csv.reader(f)
            for i, row in enumerate(data):
                if i == self.args.data_number:
                    break
                else:
                    tp_need.append(float(row[0]))
                    rtt_need.append(float(row[1]))
                    application_number.append(int(row[2]))
                    rssi_access_point_0.append(float(row[3]))
                    rssi_access_point_1.append(float(row[4]))
                    rssi_access_point_2.append(float(row[5]))
                    combination_terminal_ap.append(int(row[6]))

        return np.array(tp_need), np.array(rtt_need), np.array(application_number), np.array(rssi_access_point_0), np.array(rssi_access_point_1), np.array(rssi_access_point_2), np.array(combination_terminal_ap)

class Dataset_Split(Dataset):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = [int(i) for i in index]
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, item):
        data, label = self.dataset[self.index[item]]
        return data.clone().detach(), label.clone().detach()
    
class Application_Information_Normal():
    def __init__(self, args, application_set):
        self.args = args
        self.application_set = application_set

    def application_information_random_data(self):
        tp_need = []
        rtt_need = []
        application_id = []

        for id in self.application_set:
            tp_need.append(application_type[id]["tp_need"])
            rtt_need.append(application_type[id]["rtt_need"])
            application_id.append(id)
        
        tp_need = np.array(tp_need)
        rtt_need = np.array(rtt_need)
        tp_need_norm = self.standardization(tp_need)
        rtt_need_norm = self.standardization(rtt_need)
        network_indicator = torch.stack([tp_need_norm, rtt_need_norm], axis = 1)

        application_number = self.args.application_number
        application_id = torch.tensor(application_id)
        application_id_encoded = one_hot(application_id % application_number)
        feature_set = torch.cat([network_indicator, application_id_encoded], 1).float()

        return feature_set
    
    def standardization(self, data):
        train_data_std = (data - np.mean(data)) / np.std(data)
        return torch.from_numpy(train_data_std).float()
    
class Application_Information_Propagation_Loss(Application_Information_Normal):
    def __init__(self, args, application_set, fitted_kpca=None):
        super().__init__(args, application_set)
        self.rssi_all = self.fspl_rssi() if self.args.propagation_loss == 'fspl' else self.nfspl_rssi()
        self.fitted_kpca = fitted_kpca
    
    def application_information_random_data(self):
        tp_need = []
        rtt_need = []
        application_id = []
        rssi_access_point_0 = []
        rssi_access_point_1 = []
        rssi_access_point_2 = []

        for index, id in enumerate(self.application_set):
            tp_need.append(application_type[id]["tp_need"])
            rtt_need.append(application_type[id]["rtt_need"])
            application_id.append(id)
            rssi_access_point_0.append(self.rssi_all[index][0])
            rssi_access_point_1.append(self.rssi_all[index][1])
            rssi_access_point_2.append(self.rssi_all[index][2])
        
        tp_need = np.array(tp_need)
        rtt_need = np.array(rtt_need)
        tp_need_norm = self.standardization(tp_need)
        rtt_need_norm = self.standardization(rtt_need)
        network_indicator = torch.stack([tp_need_norm, rtt_need_norm], axis = 1)

        application_number = self.args.application_number
        application_id = torch.tensor(application_id)
        application_id_encoded = one_hot(application_id % application_number)
        application_information = torch.cat([network_indicator, application_id_encoded], 1).float()

        rssi_access_point_0 = np.array(rssi_access_point_0)
        rssi_access_point_1 = np.array(rssi_access_point_1)
        rssi_access_point_2 = np.array(rssi_access_point_2)
        rssi_access_point_0_norm = self.standardization(rssi_access_point_0)
        rssi_access_point_1_norm = self.standardization(rssi_access_point_1)
        rssi_access_point_2_norm = self.standardization(rssi_access_point_2)

        if self.args.kpca:
            rssi_stacked = torch.stack([rssi_access_point_0_norm, rssi_access_point_1_norm, rssi_access_point_2_norm], axis=1)
            rssi_kpca_np = self.fitted_kpca.transform(rssi_stacked.numpy())

            rssi_kpca = torch.tensor(rssi_kpca_np, dtype=torch.float32)
            feature_set = torch.cat([application_information, rssi_kpca], dim=1)

        else:
            rssi_access_point_0_norm = rssi_access_point_0_norm.unsqueeze(1)
            rssi_access_point_1_norm = rssi_access_point_1_norm.unsqueeze(1)
            rssi_access_point_2_norm = rssi_access_point_2_norm.unsqueeze(1)

            stack = torch.cat([application_information, rssi_access_point_0_norm], dim = 1)
            stack = torch.cat([stack, rssi_access_point_1_norm], dim = 1)
            feature_set = torch.cat([stack, rssi_access_point_2_norm], dim = 1)

        return feature_set

    def fspl_rssi(self):
        tssi = 20

        rssi_all = []
        access_point_position_set = []
        for id in range(self.args.access_point_number):
            x = 30 * (id + 1)
            access_point_position_set.append((x, 0))

        terminal_position_set = []
        for id in range(self.args.unlabel_data_number):
            x_ranges = [(0, 29), (32, 59), (62, 89), (92, 121)]
            x_selected_range = random.choice(x_ranges)
            x_float = random.uniform(*x_selected_range)
            y_ranges = [(-30, -1), (2, 31)]
            y_selected_range = random.choice(y_ranges)
            y_float = random.uniform(*y_selected_range)
            terminal_position_set.append((x_float, y_float))

        rssi_all = []
        for terminal_position in terminal_position_set:
            rssi_set = []
            for access_point_position in access_point_position_set:
                distance = self.distance_calculation(access_point_position, terminal_position)
                fspl = self.fspl_calculation(distance)
                rssi = self.fspl_rssi_calculation(fspl, tssi)
                rssi_set.append(rssi)
            rssi_all.append(rssi_set)
        
        return rssi_all

    def nfspl_rssi(self):
        tssi = 20

        rssi_all = []
        access_point_position_set = []
        for id in range(self.args.access_point_number):
            x = 30 * (id + 1)
            access_point_position_set.append((x, 0))

        terminal_position_set = []
        for id in range(self.args.unlabel_data_number):
            x_ranges = [(0, 29), (32, 59), (62, 89), (92, 121)]
            x_selected_range = random.choice(x_ranges)
            x_float = random.uniform(*x_selected_range)
            y_ranges = [(-30, -16), (-13, -1), (2, 14), (17, 31)]
            y_selected_range = random.choice(y_ranges)
            y_float = random.uniform(*y_selected_range)
            terminal_position_set.append((x_float, y_float))

        rssi_all = []
        for terminal_position in terminal_position_set:
            rssi_set = []
            for access_point_position in access_point_position_set:
                distance = self.distance_calculation(access_point_position, terminal_position)
                fspl = self.fspl_calculation(distance)
                wall_loss = self.wall_loss_calculation(terminal_position)
                rssi = self.nfspl_rssi_calculation(fspl, wall_loss, tssi)
                rssi_set.append(rssi)
            rssi_all.append(rssi_set)
        
        return rssi_all
        
    def distance_calculation(self, access_point_position, terminal_position):
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(access_point_position, terminal_position)))

    def fspl_calculation(self, distance):
        return 20 * math.log10(4 * math.pi * distance * 5.0e9 / 299792458)

    def wall_loss_calculation(self, terminal_position):
        wall_loss = 20
        if terminal_position[1] <= -17 or 17 <= terminal_position[1]:
            return wall_loss
        else:
            return 0

    def fspl_rssi_calculation(self, fspl, tssi):
        return tssi - fspl
    
    def nfspl_rssi_calculation(self, fspl, wall_loss, tssi):
        return tssi - fspl - wall_loss