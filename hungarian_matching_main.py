from os.path import dirname, abspath
import sys
from option import args_parser
from access_point import create_access_point
from terminal import create_terminal
import rand as rand
from access_point import network_indicator_connected
from propagation_loss import Propagation_Loss
from terminal_satisfaction import Terminal_Satisfaction
from terminal_satisfaction import Terminal_Satisfaction_Propagation_Loss
from terminal_satisfaction import Terminal_Satisfaction_After_Allocation
from terminal_satisfaction import Terminal_Satisfaction_After_Allocation_Propagation_Loss
from hungarian import Hungarian_Propagation_Loss
import csv
import time

parent_directory = dirname(abspath(__file__))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

if __name__ == '__main__':

    args = args_parser()
    access_point_set = create_access_point(args.access_point_number)
    terminal_set = create_terminal(args)

    previous_satisfaction_harmonic_mean = []
    previous_satisfaction_harmonic_mean_fspl = []
    previous_satisfaction_harmonic_mean_nfspl = []
    satisfaction_harmonic_mean = []
    satisfaction_harmonic_mean_fspl = []
    satisfaction_harmonic_mean_nfspl = []

    rssi_all = []
    application_id_set = []
    application_tp_need_set = []
    application_rtt_need_set = []
    objective_value = []
    for i in range(args.simulation_number):
        print("-------------------------------------------------------------")
        print(f'Number of allocations = No. {str(i+1)}\n')
        start = time.time()
              
        rand.application_random(args, terminal_set)
        rand.access_point_random(args, terminal_set)

        network_indicator_connected(args,access_point_set, terminal_set)

        terminal_satisfaction = Terminal_Satisfaction(args, access_point_set, terminal_set)
        previous_satisfaction_harmonic_mean.append(terminal_satisfaction.terminal_satisfaction())
        print(f'\nTerminal Satisfaction Normal: {round(previous_satisfaction_harmonic_mean[i], 4)}')

        """
        propagation_loss = Propagation_Loss(access_point_set, terminal_set)
        rssi, fspl = propagation_loss.free_space_path_loss()
        for item in rssi:
            rssi_all.append(item)
        terminal_satisfaction_fspl = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, fspl)
        previous_satisfaction_harmonic_mean_fspl.append(terminal_satisfaction_fspl.terminal_satisfaction())
        print(f'Terminal Satisfaction FSPL: {round(previous_satisfaction_harmonic_mean_fspl[i], 4)}')
        """

        propagation_loss = Propagation_Loss(access_point_set, terminal_set)
        rssi, nfspl = propagation_loss.non_free_space_path_loss()
        for item in rssi:
            rssi_all.append(item)
        terminal_satisfaction_nfspl = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, nfspl)
        previous_satisfaction_harmonic_mean_nfspl.append(terminal_satisfaction_nfspl.terminal_satisfaction())
        print(f'Terminal Satisfaction NFSPL: {round(previous_satisfaction_harmonic_mean_nfspl[i], 4)}')

        print('\nAfter allocation ↓\n')

        if args.learning_method == 'hungarian':
            '''
            hungarian_normal = Hungarian(args, access_point_set, terminal_set)
            combination_set = hungarian_normal.hungarian()
            terminal_satisfaction = Terminal_Satisfaction_After_Allocation(args, access_point_set, terminal_set, combination_set)
            terminal_satisfaction_normal = terminal_satisfaction.terminal_satisfaction_after_allocation()
            satisfaction_harmonic_mean.append(terminal_satisfaction_normal)
            print(f'Terminal Satisfaction Normal: {round(terminal_satisfaction_normal, 4)}')
            '''
            
            '''
            hungarian_fspl = Hungarian_Propagation_Loss(args, access_point_set, terminal_set, fspl)
            combination_set = hungarian_fspl.hungarian()
            terminal_satisfaction = Terminal_Satisfaction_After_Allocation_Propagation_Loss(args, access_point_set, terminal_set, fspl, combination_set)
            terminal_satisfaction_fspl = terminal_satisfaction.terminal_satisfaction_after_allocation()
            satisfaction_harmonic_mean.append(terminal_satisfaction_fspl)
            print(f'Terminal Satisfaction FSPL: {round(terminal_satisfaction_fspl, 4)}')

            for index, terminal in enumerate(terminal_set):
                application_tp_need_set.append(terminal.tp_need)
                application_rtt_need_set.append(terminal.rtt_need)
                application_id_set.append(terminal.application_id)
                objective_value.append(combination_set[index])
            '''

            hungarian_nfspl = Hungarian_Propagation_Loss(args, access_point_set, terminal_set, nfspl)
            combination_set = hungarian_nfspl.hungarian()
            terminal_satisfaction = Terminal_Satisfaction_After_Allocation_Propagation_Loss(args, access_point_set, terminal_set, nfspl, combination_set)
            terminal_satisfaction_nfspl = terminal_satisfaction.terminal_satisfaction_after_allocation()
            satisfaction_harmonic_mean.append(terminal_satisfaction_nfspl)
            print(f'Terminal Satisfaction NFSPL: {round(terminal_satisfaction_nfspl, 4)}')

            for index, terminal in enumerate(terminal_set):
                application_tp_need_set.append(terminal.tp_need)
                application_rtt_need_set.append(terminal.rtt_need)
                application_id_set.append(terminal.application_id)
                objective_value.append(combination_set[index])

        end = time.time()
        print(f'Execution time: {round(end - start, 6)}')

    supervised_data = []
    for index in range(len(application_tp_need_set)):
        row = []
        row.append(application_tp_need_set[index])
        row.append(application_rtt_need_set[index])
        row.append(application_id_set[index])
        for rssi in rssi_all[index]:
            row.append(round(rssi, 3))
        row.append(objective_value[index])
        supervised_data.append(row)
    
    file_name = str(args.terminal_number) + '_terminal_nfspl.csv'
    with open(file_name, 'a') as f:
        writer_object = csv.writer(f)
        for item in supervised_data:
            writer_object.writerow(item)
        f.close()