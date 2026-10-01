import csv
import numpy as np

class Greedy_Algorithm:
    def __init__(self, args, terminal_set):
        self.args = args
        self.terminal_set = terminal_set
        self.file_name = str(self.args.terminal_number) + '_terminal_' + str(args.propagation_loss) + '.csv'
        self.supervised_dataset = self.supervised_data()
        self.dataset = self.dataset_random()
        
    def supervised_data(self):
        dataset = []
        with open(self.file_name, 'r') as f:
            data = csv.reader(f)
            for index, row in enumerate(data):
                if index == 2519:
                    break
                else:
                    dataset.append(row)

        for index, item in enumerate(dataset):
            dataset[index] = [int(element) for element in item]

        amount = max(2519, 1)
        dataset_index = np.random.choice(range(2519), amount, replace = False)

        dataset_new = np.array([[0, 0, 0, 0]]) if self.args.propagation_loss == 'normal' else np.array([[0, 0, 0, 0, 0, 0, 0]])

        for index in dataset_index:
            dataset_new = np.vstack([dataset_new, dataset[index]])
        dataset = dataset_new.tolist()

        return dataset

    def dataset_random(self):
        tp_need = []
        rtt_need = []
        application_id = []
        for terminal in self.terminal_set:
            tp_need.append(terminal.tp_need)
            rtt_need.append(terminal.rtt_need)
            application_id.append(terminal.application_id)
        dataset_random = np.array(list(zip(tp_need, rtt_need, application_id)))
        return dataset_random

    def greedy(self):
        access_point_prediction = []
        sorted_data = sorted(self.supervised_dataset, key=lambda row: (row[0], -row[1], -row[2]), reverse=True)
        for data in self.dataset:
            access_point_prediction.append(self.greedy_calculation(data, sorted_data))
        
        for index, terminal in enumerate(self.terminal_set):
            terminal.access_point_id_set(access_point_prediction[index])
        
        return access_point_prediction
    
    def greedy_calculation(self, data, sorted_data):
        best_row = None
        for row in sorted_data:
            if row[0] == data[0] and row[1] == data[1] and row[2] == data[2]:
                best_row = row
                break
        
        if best_row:
            return best_row[3]
        else:
            return "No matching access point found"