from os.path import dirname, abspath
import sys
import os
from tensorboardX import SummaryWriter
from option import args_parser
from utils import exp_details
import time
from tqdm import tqdm

from access_point import create_access_point
from terminal import create_terminal
import rand
from access_point import network_indicator_connected
from propagation_loss import Propagation_Loss
from terminal_satisfaction import Terminal_Satisfaction
from terminal_satisfaction import Terminal_Satisfaction_Propagation_Loss
from terminal_satisfaction import Terminal_Satisfaction_After_Allocation
from greedy import Greedy_Algorithm
import time
from figure import File_Save

parent_directory = dirname(abspath(__file__))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

if __name__ == '__main__':
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    satis_harmean_simulation = []
    learning_time_simulation = []
    allocation_time_simulation = []
    for simulation in tqdm(range(args.simulation_number)):
        learning_time_simulation.append(0)

        access_point_set = create_access_point(args.access_point_number)
        terminal_set = create_terminal(args.terminal_number)

        previous_satisfaction_harmonic_mean = []
        previous_satisfaction_harmonic_mean_fspl = []
        previous_satisfaction_harmonic_mean_nfspl = []
        satisfaction_harmonic_mean = []
        allocation_time = []

        distance_all = []
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

            network_indicator_connected(access_point_set, terminal_set)

            if args.propagation_loss == 'normal':
                terminal_satisfaction = Terminal_Satisfaction(args, access_point_set, terminal_set)
                previous_satisfaction_harmonic_mean.append(terminal_satisfaction.terminal_satisfaction())
                print(f'\nTerminal Satisfaction Normal: {round(previous_satisfaction_harmonic_mean[i], 4)}')

            elif args.propagation_loss == 'fspl':
                propagation_loss = Propagation_Loss(access_point_set, terminal_set)
                distance, fspl = propagation_loss.free_space_path_loss()
                for item in distance:
                    distance_all.append(item)
                terminal_satisfaction_fspl = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, fspl)
                previous_satisfaction_harmonic_mean_fspl.append(terminal_satisfaction_fspl.terminal_satisfaction())
                print(f'Terminal Satisfaction FSPL: {round(previous_satisfaction_harmonic_mean_fspl[i], 4)}')

            elif args.propagation_loss == 'nfspl':
                nfspl = propagation_loss.non_free_space_path_loss()
                terminal_satisfaction_nfspl = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, nfspl)
                previous_satisfaction_harmonic_mean_nfspl.append(terminal_satisfaction_nfspl.terminal_satisfaction())
                print(f'Terminal Satisfaction NFSPL: {previous_satisfaction_harmonic_mean_nfspl[i]}')

            print('\nAfter allocation ↓\n')
            
            greedy_algorithm = Greedy_Algorithm(args, terminal_set)
            combination_set = greedy_algorithm.greedy()
            terminal_satisfaction = Terminal_Satisfaction_After_Allocation(args, access_point_set, terminal_set, combination_set)
            terminal_satisfaction_normal = terminal_satisfaction.terminal_satisfaction_after_allocation()
            satisfaction_harmonic_mean.append(terminal_satisfaction_normal)
            print(f'Terminal Satisfaction Normal: {round(terminal_satisfaction_normal, 4)}')

            end = time.time()
            allocation_time.append(round(end - start, 6))
            print(f'Execution time: {round(end - start, 6)}')
        
        satis_harmean_simulation.append(sum(satisfaction_harmonic_mean) / len(satisfaction_harmonic_mean))
        allocation_time_simulation.append(sum(allocation_time) / len(allocation_time))
    
    if args.simulation_number == 100:
        file = File_Save(args)
        file.save(learning_time_simulation, 'learning_time')
        file.save(allocation_time_simulation, 'allocation_time')
        file.save(satis_harmean_simulation, 'terminal_satisfaction')