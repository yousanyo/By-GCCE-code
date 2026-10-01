import matplotlib.pyplot as plt
import csv
import os
from option import args_parser
from csv import writer

args = args_parser()

class File_Save():
    def __init__(self, args):
        self.args = args
        
    def save(self, element_set, element_name):
        if self.args.learning_method == 'hfl':
            file_name = str(self.args.terminal_number) + '_' + element_name + '_' + self.args.learning_method + '_' + self.args.update_method + '_' + self.args.optimizer + '_' + self.args.propagation_loss + '_α_' + str(self.args.alpha) + '_global_epoch_' + str(self.args.global_epoch) + '_mid_epoch_' + str(self.args.mid_epoch) + '_local_epoch_' + str(self.args.local_epoch) + '.csv'
        elif self.args.learning_method == 'fl':
            file_name = str(self.args.terminal_number) + '_' + element_name + '_' + self.args.learning_method + '_' + self.args.update_method + '_' + self.args.optimizer + '_' + self.args.propagation_loss + '_α_' + str(self.args.alpha) + '_global_epoch_' + str(self.args.global_epoch) + '_local_epoch_' + str(self.args.local_epoch) + '.csv'
        elif self.args.learning_method == 'cl':
            file_name = str(self.args.terminal_number) + '_' + element_name + '_' + self.args.learning_method + '_' + self.args.optimizer + '_' + self.args.propagation_loss + '_global_epoch_' + str(self.args.global_epoch) + '.csv'
        elif self.args.learning_method == 'greedy':
            file_name = str(self.args.terminal_number) + '_' + element_name + '_' + self.args.learning_method + '_' + self.args.propagation_loss + '.csv'
        folder_path = 'dataset/' + element_name
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w') as f:
            writer_object = writer(f)
            for item in element_set:
                writer_object.writerow([item])
            f.close()

class File_Save_Propagation_Loss():
    def __init__(self, args):
        self.args = args
        
    def save(self, element_set, element_name, propagation_loss):
        kpca = '_kpca' if self.args.kpca else ''
        if self.args.learning_method == 'hfl':
            file_name = self.args.propagation_loss + '_model_' + self.args.learning_method + '_' + self.args.update_method + kpca  + '_ge_' + str(self.args.global_epoch) + '_me_' + str(self.args.mid_epoch) + '_le_' + str(self.args.local_epoch) + '.csv'
        elif self.args.learning_method == 'fl':
            file_name = self.args.propagation_loss + '_model_' + self.args.learning_method + '_' + self.args.update_method + kpca + '_ge_' + str(self.args.global_epoch) + '_le_' + str(self.args.local_epoch) + '.csv'
        elif self.args.learning_method == 'cl':
            file_name = self.args.propagation_loss + '_model_' + self.args.learning_method + kpca + '_ge_' + str(self.args.global_epoch) + '.csv'
        elif self.args.learning_method == 'greedy':
            file_name = str(self.args.terminal_number) + '_' + element_name + '_' + self.args.learning_method + '_' + self.args.propagation_loss + '.csv'
        
        self.args.dirichlet
        if element_name == 'terminal_satisfaction':
            folder_path = 'dataset_propagation_loss/' + 'α_' + str(self.args.alpha) + '/' + self.args.dirichlet + '/' + element_name + '_' + propagation_loss + '_loss'
        else:
            folder_path = 'dataset_propagation_loss/' + 'α_' + str(self.args.alpha) + '/' + self.args.dirichlet + '/' + element_name
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w') as f:
            writer_object = writer(f)
            for item in element_set:
                writer_object.writerow([item])
            f.close()

class File_Save_Propagation_Loss_Global_Epoch():
    def __init__(self, args):
        self.args = args
        
    def save(self, element_set, element_name, propagation_loss):
        if self.args.learning_method == 'hfl':
            if element_name == 'terminal_satisfaction':
                file_name = self.args.propagation_loss + '_model_' + self.args.learning_method + '_' + self.args.update_method + '_' + propagation_loss + '_propagation_loss_α_' + str(self.args.alpha) + '_global_epoch_' + str(self.args.global_epoch) + '_mid_epoch_' + str(self.args.mid_epoch) + '_local_epoch_' + str(self.args.local_epoch) + '.csv'
            else:
                file_name = self.args.propagation_loss + '_model_' + self.args.learning_method + '_' + self.args.update_method + '_α_' + str(self.args.alpha) + '_global_epoch_' + str(self.args.global_epoch) + '_local_epoch_' + str(self.args.local_epoch) + '.csv'
        elif self.args.learning_method == 'fl':
            if element_name == 'terminal_satisfaction':
                file_name = self.args.propagation_loss + '_model_' + self.args.learning_method + '_' + self.args.update_method + '_' + propagation_loss + '_propagation_loss_α_' + str(self.args.alpha) + '_global_epoch_' + str(self.args.global_epoch) + '_local_epoch_' + str(self.args.local_epoch) + '.csv'
            else:
                file_name = self.args.propagation_loss + '_model_' + self.args.learning_method + '_' + self.args.update_method + '_α_' + str(self.args.alpha) + '_global_epoch_' + str(self.args.global_epoch) + '_local_epoch_' + str(self.args.local_epoch) + '.csv'
        elif self.args.learning_method == 'cl':
            file_name = self.args.propagation_loss + '_' + str(self.args.terminal_number) + '_' + element_name + '_' + self.args.learning_method + '_' + self.args.optimizer + '_' + propagation_loss + '_global_epoch_' + str(self.args.global_epoch) + '.csv'
        elif self.args.learning_method == 'greedy':
            file_name = str(self.args.terminal_number) + '_' + element_name + '_' + self.args.learning_method + '_' + self.args.propagation_loss + '.csv'
    
        if element_name == 'terminal_satisfaction':
            folder_path = 'dataset_global_epoch/' + element_name + '_' + self.args.propagation_loss
        else:
            folder_path = 'dataset_global_epoch/' + element_name
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w') as f:
            writer_object = writer(f)
            for item in element_set:
                writer_object.writerow([item])
            f.close()

def processing_time(terminal_number, learning_method, update_method, optimizer, propagation_loss, best_model, alpha, global_epoch, mid_epoch, local_epoch):
    file_name_learning_time = 'dataset/learning_time/' + str(terminal_number) + '_learning_time_' + learning_method + '_' + update_method + '_' + optimizer + '_' + propagation_loss + '_best_model_' + str(best_model) + '_α_' + str(alpha) + '_global_epoch_' + str(global_epoch) + '_mid_epoch_' + str(mid_epoch) + '_local_epoch_' + str(local_epoch) + '.csv'
    with open(file_name_learning_time, 'r') as file:
        reader = csv.reader(file)
        learning = [float(row[0]) for row in reader]

    file_name_allocation_time = 'dataset/allocation_time/' + str(terminal_number) + '_allocation_time_' + learning_method + '_' + update_method + '_' + optimizer + '_' + propagation_loss + '_best_model_' + str(best_model) + '_α_' + str(alpha) + '_global_epoch_' + str(global_epoch) + '_mid_epoch_' + str(mid_epoch) + '_local_epoch_' + str(local_epoch) + '.csv'
    with open(file_name_allocation_time, 'r') as file:
        reader = csv.reader(file)
        allocation = [float(row[0]) for row in reader]

    processing_time = [(float(learning[index]) + float(allocation[index])) for index in range(len(learning))]

    return processing_time

def terminal_satisfaction(terminal_number, learning_method, update_method, optimizer, propagation_loss, best_model, alpha, global_epoch, mid_epoch, local_epoch):
    file_name_learning_time = 'dataset/terminal_satisfaction/' + str(terminal_number) + '_terminal_satisfaction_' + learning_method + '_' + update_method + '_' + optimizer + '_' + propagation_loss + '_best_model_' + str(best_model) + '_α_' + str(alpha) + '_global_epoch_' + str(global_epoch) + '_mid_epoch_' + str(mid_epoch) + '_local_epoch_' + str(local_epoch) + '.csv'
    with open(file_name_learning_time, 'r') as file:
        reader = csv.reader(file)
        terminal_satisfaction = [float(row[0]) for row in reader]

    return terminal_satisfaction

def plot(result_value, result_label, color, simulation_number, y_label, y_lim_left, y_lim_right, legend_location, anchor_left, anchor_right, file_name):
    fig, ax = plt.subplots()
    ax.set_facecolor((0.83, 0.83, 0.83, 0.1))

    x_values = range(1, simulation_number + 1)
    for i in range(len(result_value)):
        plt.plot(x_values, result_value[i], label=str(result_label[i]), color=color[i], linewidth=2, marker='^', markevery=range(9, simulation_number, 10), markersize=7, alpha=0.7)
    plt.xlabel('Simulation number', fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.xlim(1, simulation_number + 1)
    plt.ylim(y_lim_left, y_lim_right)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend(loc=legend_location, bbox_to_anchor=(anchor_left, anchor_right), fontsize=14)
    plt.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5, color='gray')

    file_path = os.path.join('figure', file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    if args.best:
        best_model = 'True'
    else:
        best_model = 'False'

    processing_time_result = []
    processing_time_label = ['FedAVG', 'FedProx', 'Scaffold', 'FedNova', 'FedDF', 'MOON']
    color = ['r', 'g', 'b', 'y', 'm']
    file_name = str(args.terminal_number) + '_processing_time_' + args.learning_method + '_' + args.optimizer + '_' + args.propagation_loss + '_best_model_' + best_model + '_α_' + str(args.alpha) + '_global_epoch_' + str(args.global_epoch) + '_mid_epoch_' + str(args.mid_epoch) + '_local_epoch_' + str(args.local_epoch) + '.png'
    processing_time_result.append(processing_time(args.terminal_number, args.learning_method, 'fedavg', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    processing_time_result.append(processing_time(args.terminal_number, args.learning_method, 'fedprox', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    processing_time_result.append(processing_time(args.terminal_number, args.learning_method, 'scaffold', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    processing_time_result.append(processing_time(args.terminal_number, args.learning_method, 'fednova', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    processing_time_result.append(processing_time(args.terminal_number, args.learning_method, 'feddf', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    processing_time_result.append(processing_time(args.terminal_number, args.learning_method, 'moon', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    plot(processing_time_result, processing_time_label, color, args.simulation_number, 'Processing time (second)', 0, 12.0, 'upper left', 0.36, 0.45, file_name)

    terminal_satisfaction_result = []
    terminal_satisfaction_label = ['FedAVG', 'FedProx', 'Scaffold', 'FedNova', 'FedDF', 'MOON']
    # terminal_satisfaction_label = ['FedDF']
    color = ['r', 'g', 'b', 'y', 'm']
    # color = ['m']
    file_name = str(args.terminal_number) + '_terminal_satisfaction_' + args.learning_method + '_' + args.optimizer + '_' + args.propagation_loss + '_best_model_' + best_model + '_α_' + str(args.alpha) + '_global_epoch_' + str(args.global_epoch) + '_mid_epoch_' + str(args.mid_epoch) + '_local_epoch_' + str(args.local_epoch) + '.png'
    terminal_satisfaction_result.append(terminal_satisfaction(args.terminal_number, args.learning_method, 'fedavg', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    terminal_satisfaction_result.append(terminal_satisfaction(args.terminal_number, args.learning_method, 'fedprox', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    terminal_satisfaction_result.append(terminal_satisfaction(args.terminal_number, args.learning_method, 'scaffold', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    terminal_satisfaction_result.append(terminal_satisfaction(args.terminal_number, args.learning_method, 'fednova', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    terminal_satisfaction_result.append(terminal_satisfaction(args.terminal_number, args.learning_method, 'feddf', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    terminal_satisfaction_result.append(terminal_satisfaction(args.terminal_number, args.learning_method, 'moon', args.optimizer, args.propagation_loss, best_model, args.alpha, args.global_epoch, args.mid_epoch, args.local_epoch))
    plot(terminal_satisfaction_result, terminal_satisfaction_label, color, args.simulation_number, 'Terminal satisfaction', -0.19, 0.9, 'lower right', 1, 0, file_name)
    