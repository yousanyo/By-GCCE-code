import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--terminal_number', type = int, default = 100, help = 'Number of terminals')
    parser.add_argument('--application_number', type = int, default = 4, help = 'Number of applications')
    parser.add_argument('--access_point_number', type = int, default = 3, help = 'Number of access points')
    parser.add_argument('--mid_server', type = list, default = [3], help = 'Number of mid_server')
    parser.add_argument('--user_fraction', type = float, default = 0.2, help = 'Fraction of the participating user')
    parser.add_argument('--local_batch_size', type = int, default = 32, help = 'Local batch size')
    parser.add_argument('--unlabel_data_number', type = int, default = 128, help = 'Unlabel data number for FedDF')
    parser.add_argument('--best', type = bool, default = True, help = 'Use of best model')
    parser.add_argument('--download', type = bool, default = True, help = 'Download of learning model')
    parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'Optimizer')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate')
    parser.add_argument('--scheduler', type = bool, default = False, help='Scheduler of LR')
    parser.add_argument('--hidden_number', type = int, default = 128, help = 'Number of hidden layers')
    parser.add_argument('--gpu', default = 0, help = 'To use cuda, set to a specific GPU ID. Default set to use CPU.')

    # fedprox
    parser.add_argument('--mu_fedprox', type = float, default = 0.01, help='Proximal term constant for fedprox(0.01 is the default)')

    # FedNova
    parser.add_argument('--rho', type = int, default = 0.9, help = 'Rho in FedNova')

    # FedDF
    parser.add_argument('--t_feddf', type = float, default = 15, help = 'Temperature for Kullback-Leibler (KL) Divergence(15 is the default)')

    # MOON
    parser.add_argument('--mu_moon', type = float, default = 0.1, help='Proximal term constant for moon(0.1 is the default)')
    parser.add_argument('--t_moon', type = float, default = 0.01, help = 'Temperature for moon(0.01 is the default)')
    parser.add_argument('--model_buffer_size', type = int, default = 1, help = 'Previous model buffer size for moon')
    parser.add_argument('--previous_model_pool_option', type = str, default = 'FIFO', help = 'Previous model pool method for moon')
    
    parser.add_argument('--all_path_loss', type = bool, default = True, help = 'Condition for all path loss scenario')
    parser.add_argument('--simulation_number', type = int, default = 100, help = 'Number of simulations')
    parser.add_argument('--allocation_number', type = int, default = 100, help = 'Number of allocations')
    parser.add_argument('--global_epoch', type = int, default = 10, help = 'Number of global epochs')
    parser.add_argument('--mid_epoch', type = int, default = 1, help = 'Number of mid-epochs')
    parser.add_argument('--local_epoch', type = int, default = 10, help = 'Number of local epochs')
    parser.add_argument('--data_number', type = int, default = 8750, help = 'Number of supervised data')
    parser.add_argument('--min_required_size', type = int, default = 35, help = 'Min required dataset size for quantity distribution skew')
    parser.add_argument('--max_required_size', type = int, default = 280, help = 'Max required dataset size for quantity distribution skew')

    parser.add_argument('--dirichlet', type = str, default = 'label', help = 'Type of dirichlet distribution')
    parser.add_argument('--alpha', type = float, default = 1.0, help = 'Alpha of dirichlet distribution')
    parser.add_argument('--kpca', type = bool, default = False, help = 'Use of kpca')
    parser.add_argument('--learning_method', type = str, default = 'hungarian', help = 'Learning method')
    parser.add_argument('--update_method', type = str, default = 'fedavg', help = 'Local update method')
    parser.add_argument('--propagation_loss', type = str, default = 'fspl', help = 'Condition of path loss')

    args = parser.parse_args()
    return args