"""
Unified main entry point for all learning methods:
  --learning_method fl        : Federated Learning
  --learning_method hfl       : Hierarchical Federated Learning
  --learning_method cl        : Centralized Learning
  --learning_method greedy    : Greedy Algorithm (no learning)

Combine with --all_path_loss to evaluate all three propagation-loss
types (normal / fspl / nfspl) in a single run, mirroring the behaviour
of the *_all_path_loss_main.py variants.

All other arguments are forwarded to the existing args_parser() as usual.
"""

from os.path import dirname, abspath
import sys

parent_dir = dirname(abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# ── Standard library ──────────────────────────────────────────────────────────
import copy
import math
import random
import time

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project: configuration & utilities ───────────────────────────────────────
from option import args_parser
from utils import exp_details, average_weight, average_weight_scaffold, average_weight_fednova

# ── Project: datasets ─────────────────────────────────────────────────────────
from dataset import (
    Supervised_Data_Normal,
    Supervised_Data_Propagation_Loss,
    Application_Information_Normal,
    Application_Information_Propagation_Loss,
    Feature_Random_Normal,
    Feature_Random_Propagation_Loss,
)

# ── Project: models ───────────────────────────────────────────────────────────
from learning_model import (
    MLP, MLP_Scaffold, MLP_FedNova, MLP_Moon,
    mlp, mlp_moon,
)

# ── Project: sampling ─────────────────────────────────────────────────────────
from sampling import (
    noniid_dirichlet_label,
    noniid_dirichlet_feature,
    noniid_dirichlet_quantity,
    noniid_dirichlet_label_feature,
    noniid_dirichlet_label_quantity,
    noniid_dirichlet_feature_quantity,
    noniid_dirichlet_label_feature_quantity
)

# ── Project: local update / inference ─────────────────────────────────────────
from local_update import (
    Local_Update, Local_Update_FedProx, Local_Update_Scaffold,
    Local_Update_FedNova, Local_Update_MOON,
    test_inference, test_inference_moon,
)

# ── Project: HFL hierarchy ────────────────────────────────────────────────────
from hierarchy import Structure, Structure_Scaffold, Structure_FedNova, Structure_FedDF, Structure_MOON

# ── Project: evaluation ───────────────────────────────────────────────────────
from evaluation_function import Evaluation_Function

# ── Project: network / allocation ─────────────────────────────────────────────
from access_point import create_access_point, network_indicator_connected
from terminal import create_terminal
import rand as rand
from propagation_loss import Propagation_Loss
from terminal_satisfaction import (
    Terminal_Satisfaction,
    Terminal_Satisfaction_Propagation_Loss,
    Terminal_Satisfaction_After_Allocation,
    Terminal_Satisfaction_After_Allocation_Propagation_Loss,
)
from greedy import Greedy_Algorithm
from figure import File_Save, File_Save_Propagation_Loss


# ══════════════════════════════════════════════════════════════════════════════
# Helper: load supervised dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_supervised_dataset(args):
    """Return (train_dataset, test_dataset, true_test_label, trained_kpca) for the given propagation loss."""
    pl = args.propagation_loss
    if pl == 'normal':
        file_name = str(args.terminal_number) + '_terminal_normal.csv'
        dataset_instance = Supervised_Data_Normal(args, file_name)
    elif pl == 'fspl':
        file_name = str(args.terminal_number) + '_terminal_fspl.csv'
        dataset_instance = Supervised_Data_Propagation_Loss(args, file_name)
    elif pl == 'nfspl':
        file_name = str(args.terminal_number) + '_terminal_nfspl.csv'
        dataset_instance = Supervised_Data_Propagation_Loss(args, file_name)
    else:
        raise ValueError(f'Unknown propagation_loss: {pl}')

    dataset_outputs = dataset_instance.supervised_data()
    trained_kpca = getattr(dataset_instance, 'kpca', None)
    
    return dataset_outputs, trained_kpca

# ══════════════════════════════════════════════════════════════════════════════
# Helper: partition dataset
# ══════════════════════════════════════════════════════════════════════════════

def partition_dataset(dirichlet, train_dataset, train_application_number, min_required_size, max_required_size, terminal_number, alpha):
    """Return user_group for the given dataset distribution skew."""
    if dirichlet == 'label':
        user_group = noniid_dirichlet_label(train_dataset, terminal_number, alpha)
    elif dirichlet == 'feature':
        user_group = noniid_dirichlet_feature(train_dataset, train_application_number, terminal_number, alpha)
    elif dirichlet == 'quantity':
        user_group = noniid_dirichlet_quantity(train_dataset, min_required_size, max_required_size, terminal_number, alpha)
    elif dirichlet == 'label_feature':
        user_group = noniid_dirichlet_label_feature(train_dataset, train_application_number, terminal_number, alpha)
    elif dirichlet == 'label_quantity':
        user_group = noniid_dirichlet_label_quantity(train_dataset, min_required_size, max_required_size, terminal_number, alpha)
    elif dirichlet == 'feature_quantity':
        user_group = noniid_dirichlet_feature_quantity(train_dataset, train_application_number, min_required_size, max_required_size, terminal_number, alpha)
    elif dirichlet == 'label_feature_quantity':
        user_group = noniid_dirichlet_label_feature_quantity(train_dataset, train_application_number, min_required_size, max_required_size, terminal_number, alpha)
    else:
        raise ValueError(f'Unknown data skew: {dirichlet}')

    return user_group

# ══════════════════════════════════════════════════════════════════════════════
# Helper: build global model
# ══════════════════════════════════════════════════════════════════════════════

def build_global_model(args, input_number, device):
    """Instantiate the global model according to update_method."""
    m = args.update_method
    hidden = args.hidden_number
    out = args.access_point_number

    if m in ('fedavg', 'fedprox', 'feddf'):
        model = MLP(input_number, hidden, out)
    elif m == 'scaffold':
        model = MLP_Scaffold(input_number, hidden, out)
        for key, value in model.named_parameters():
            model.control[key]       = torch.zeros_like(value.data)
            model.delta_control[key] = torch.zeros_like(value.data)
            model.delta_y[key]       = torch.zeros_like(value.data)
    elif m == 'fednova':
        model = MLP_FedNova(input_number, hidden, out)
    elif m == 'moon':
        model = MLP_Moon(input_number, hidden, out)
    else:
        raise ValueError(f'Unknown update_method: {m}')

    model.to(device)
    model.train()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Helper: FedDF distillation step (shared by FL and HFL)
# ══════════════════════════════════════════════════════════════════════════════

def feddf_distillation(args, global_model, local_models, device, fitted_kpca):
    application_id_random_set = [
        math.floor(random.random() * args.application_number)
        for _ in range(args.unlabel_data_number)
    ]
    if args.propagation_loss == 'normal':
        unlabel_data_set = Application_Information_Normal(args, application_id_random_set).application_information_random_data()
    elif args.propagation_loss == 'fspl':
        unlabel_data_set = Application_Information_Propagation_Loss(args, application_id_random_set, fitted_kpca).application_information_random_data()
    elif args.propagation_loss == 'nfspl':
        unlabel_data_set = Application_Information_Propagation_Loss(args, application_id_random_set, fitted_kpca).application_information_random_data()

    loader = DataLoader(unlabel_data_set, args.unlabel_data_number, shuffle=False)

    server_model = copy.deepcopy(global_model)
    server_model.to(device)
    server_model.train()

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(server_model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(server_model.parameters(), lr=args.learning_rate)

    for unlabel_data in loader:
        teacher_logits_set = []
        unlabel_data = unlabel_data.to(device)
        for um in local_models:
            um.eval()
            with torch.no_grad():
                teacher_logits_set.append(um(unlabel_data))
        teacher_logits_avg = torch.mean(torch.stack(teacher_logits_set), dim=0)
        student_logits = server_model(unlabel_data)
        loss = F.kl_div(
            F.log_softmax(student_logits / args.t_feddf, dim=1),
            F.softmax(teacher_logits_avg / args.t_feddf, dim=1),
            reduction='batchmean',
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return server_model


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Inference for CL
# ══════════════════════════════════════════════════════════════════════════════

def inference(args, model, test_dataset):
    model.eval()
    total = 0.0
    correct = 0.0

    device = 'cuda' if args.gpu else 'cpu'
    test_loader = DataLoader(test_dataset, args.local_batch_size, shuffle = False)

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        output = model(data)

        _, prediction_label = torch.max(output, 1)
        prediction_label = prediction_label.view(-1)
        correct += torch.sum(torch.eq(prediction_label, label)).item()
        total += len(label)

    accuracy = correct / total
    return accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Helper: evaluation print
# ══════════════════════════════════════════════════════════════════════════════

def print_evaluation(args, global_model, test_dataset, true_test_label):
    if args.update_method == 'moon':
        test_accuracy, _, prediction_test = test_inference_moon(args, global_model, test_dataset)
    else:
        test_accuracy, _, prediction_test = test_inference(args, global_model, test_dataset)
    print(f'|---- Test Accuracy: {100 * test_accuracy:.2f}%')
    prediction_test_label = torch.tensor(prediction_test)
    ef = Evaluation_Function(args.access_point_number, prediction_test_label, true_test_label)
    ef.precision()
    ef.recall()
    ef.f1_score()
    return test_accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Helper: allocation loop (single propagation-loss mode)
# ══════════════════════════════════════════════════════════════════════════════

def run_allocation_loop(args, model_path, learning_method, device, trained_kpca=None):
    access_point_set = create_access_point(args.access_point_number)
    terminal_set     = create_terminal(args)

    satisfaction_harmonic_mean = []
    allocation_time            = []
    rssi_all               = []

    for i in range(args.allocation_number):
        print("-------------------------------------------------------------")
        print(f'Number of allocations = No. {i + 1}\n')
        start = time.time()

        rand.application_random(args, terminal_set)
        rand.access_point_random(args, terminal_set)
        network_indicator_connected(args, access_point_set, terminal_set)

        # ── Before allocation: compute satisfaction ──────────────────────────
        if args.propagation_loss == 'normal':
            ts_before = Terminal_Satisfaction(args, access_point_set, terminal_set)
            print(f'\nTerminal Satisfaction Normal (before): {round(ts_before.terminal_satisfaction(), 4)}')

        elif args.propagation_loss == 'fspl':
            pl_obj = Propagation_Loss(access_point_set, terminal_set)
            rssi, fspl = pl_obj.free_space_path_loss()
            for item in rssi:
                rssi_all.append(item)
            ts_before = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, fspl)
            print(f'Terminal Satisfaction FSPL (before): {round(ts_before.terminal_satisfaction(), 4)}')

        elif args.propagation_loss == 'nfspl':
            pl_obj = Propagation_Loss(access_point_set, terminal_set)
            rssi, nfspl = pl_obj.non_free_space_path_loss()
            for item in rssi:
                rssi_all.append(item)
            ts_before = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, nfspl)
            print(f'Terminal Satisfaction NFSPL (before): {round(ts_before.terminal_satisfaction(), 4)}')

        print('\nAfter allocation ↓\n')

        # ── After allocation ─────────────────────────────────────────────────
        if learning_method == 'greedy':
            greedy_algorithm = Greedy_Algorithm(args, terminal_set)
            combination_set  = greedy_algorithm.greedy()
            ts_after = Terminal_Satisfaction_After_Allocation(args, access_point_set, terminal_set, combination_set)
            satis = ts_after.terminal_satisfaction_after_allocation()
            satisfaction_harmonic_mean.append(satis)
            print(f'Terminal Satisfaction (after): {round(satis, 4)}')

        else:
            global_model = torch.load(model_path, weights_only=False)

            if args.propagation_loss == 'normal':
                feature      = Feature_Random_Normal(args, terminal_set)
                feature_set  = feature.application_information_random_data()
                combination_set = mlp_moon(args, global_model, feature_set) if args.update_method == 'moon' else mlp(args, global_model, feature_set)
                ts_after = Terminal_Satisfaction_After_Allocation(args, access_point_set, terminal_set, combination_set)
                satis = ts_after.terminal_satisfaction_after_allocation()
                satisfaction_harmonic_mean.append(satis)
                print(f'Terminal Satisfaction Normal: {round(satis, 4)}')

            elif args.propagation_loss == 'fspl':
                feature      = Feature_Random_Propagation_Loss(args, terminal_set, rssi_all, fitted_kpca=trained_kpca)
                feature_set  = feature.application_information_random_data()
                combination_set = mlp_moon(args, global_model, feature_set) if args.update_method == 'moon' else mlp(args, global_model, feature_set)
                ts_after = Terminal_Satisfaction_After_Allocation_Propagation_Loss(args, access_point_set, terminal_set, fspl, combination_set)
                satis = ts_after.terminal_satisfaction_after_allocation()
                satisfaction_harmonic_mean.append(satis)
                print(f'Terminal Satisfaction FSPL: {round(satis, 4)}')

            elif args.propagation_loss == 'nfspl':
                feature      = Feature_Random_Propagation_Loss(args, terminal_set, rssi_all, fitted_kpca=trained_kpca)
                feature_set  = feature.application_information_random_data()
                combination_set = mlp_moon(args, global_model, feature_set) if args.update_method == 'moon' else mlp(args, global_model, feature_set)
                ts_after = Terminal_Satisfaction_After_Allocation_Propagation_Loss(args, access_point_set, terminal_set, nfspl, combination_set)
                satis = ts_after.terminal_satisfaction_after_allocation()
                satisfaction_harmonic_mean.append(satis)
                print(f'Terminal Satisfaction NFSPL: {round(satis, 4)}')

        end = time.time()
        elapsed = round(end - start, 6)
        allocation_time.append(elapsed)
        print(f'Execution time: {elapsed}')

    return satisfaction_harmonic_mean, allocation_time


# ══════════════════════════════════════════════════════════════════════════════
# Helper: all-path-loss allocation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_allocation_loop_all_path_loss(args, model_path, trained_kpca=None):
    access_point_set = create_access_point(args.access_point_number)
    terminal_set     = create_terminal(args)

    ts_normal_set = []
    ts_fspl_set   = []
    ts_nfspl_set  = []
    allocation_time = []

    rssi_all_fspl  = []
    rssi_all_nfspl = []

    for i in range(args.allocation_number):
        print("-------------------------------------------------------------")
        print(f'Number of allocations = No. {i + 1}\n')
        start = time.time()

        rand.application_random(args, terminal_set)
        rand.access_point_random(args, terminal_set)
        network_indicator_connected(args, access_point_set, terminal_set)

        # Normal
        ts_obj = Terminal_Satisfaction(args, access_point_set, terminal_set)
        print(f'\nTerminal Satisfaction Normal (before): {round(ts_obj.terminal_satisfaction(), 4)}')

        # FSPL
        pl_obj = Propagation_Loss(access_point_set, terminal_set)
        rssi, fspl = pl_obj.free_space_path_loss()
        for item in rssi:
            rssi_all_fspl.append(item)
        ts_fspl_obj = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, fspl)
        print(f'Terminal Satisfaction FSPL (before): {round(ts_fspl_obj.terminal_satisfaction(), 4)}')

        # NFSPL
        rssi, nfspl = pl_obj.non_free_space_path_loss()
        for item in rssi:
            rssi_all_nfspl.append(item)
        ts_nfspl_obj = Terminal_Satisfaction_Propagation_Loss(args, access_point_set, terminal_set, nfspl)
        print(f'Terminal Satisfaction NFSPL (before): {round(ts_nfspl_obj.terminal_satisfaction(), 4)}')

        print('\nAfter allocation ↓\n')

        global_model = torch.load(model_path, weights_only=False)

        # Build features per propagation-loss type
        if args.propagation_loss == 'normal':
            feature = Feature_Random_Normal(args, terminal_set)
        elif args.propagation_loss == 'fspl':
            feature = Feature_Random_Propagation_Loss(args, terminal_set, rssi_all_fspl, fitted_kpca=trained_kpca)
        else:
            feature = Feature_Random_Propagation_Loss(args, terminal_set, rssi_all_nfspl, fitted_kpca=trained_kpca)

        feature_set     = feature.application_information_random_data()
        combination_set = mlp_moon(args, global_model, feature_set) if args.update_method == 'moon' else mlp(args, global_model, feature_set)

        # Evaluate under all three loss models
        ts_n = Terminal_Satisfaction_After_Allocation(args, access_point_set, terminal_set, combination_set).terminal_satisfaction_after_allocation()
        ts_f = Terminal_Satisfaction_After_Allocation_Propagation_Loss(args, access_point_set, terminal_set, fspl,  combination_set).terminal_satisfaction_after_allocation()
        ts_nf= Terminal_Satisfaction_After_Allocation_Propagation_Loss(args, access_point_set, terminal_set, nfspl, combination_set).terminal_satisfaction_after_allocation()

        ts_normal_set.append(ts_n)
        ts_fspl_set.append(ts_f)
        ts_nfspl_set.append(ts_nf)

        print(f'Terminal Satisfaction Normal: {round(ts_n, 4)}')
        print(f'Terminal Satisfaction FSPL:   {round(ts_f, 4)}')
        print(f'Terminal Satisfaction NFSPL:  {round(ts_nf, 4)}')

        end = time.time()
        elapsed = round(end - start, 6)
        allocation_time.append(elapsed)
        print(f'Execution time: {elapsed}')

    return ts_normal_set, ts_fspl_set, ts_nfspl_set, allocation_time


# ══════════════════════════════════════════════════════════════════════════════
# Learning phase: Centralized Learning
# ══════════════════════════════════════════════════════════════════════════════

def run_cl(args, device, logger, all_path_loss=False):
    print('\n======= Centralized Learning =======')

    if all_path_loss:
        ts_normal_sim = []; ts_fspl_sim = []; ts_nfspl_sim = []
        test_accuracy_sim = []; learning_time_sim = []; allocation_time_sim = []
    else:
        satis_harmean_sim = []; learning_time_sim = []; allocation_time_sim = []

    for simulation in tqdm(range(args.simulation_number)):
        start = time.time()

        dataset_outputs, _ = load_supervised_dataset(args)
        train_dataset, test_dataset, true_test_label, _ = dataset_outputs
        print(len(train_dataset), len(test_dataset))

        input_number = len(train_dataset[0][0])
        global_model = MLP(input_number, args.hidden_number, args.access_point_number)
        global_model.to(device)
        global_model.train()

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(global_model.parameters(), lr=args.learning_rate)

        train_loader    = DataLoader(train_dataset, batch_size=args.local_batch_size, shuffle=False)
        loss_function   = torch.nn.CrossEntropyLoss().to(device)

        train_loss = []

        for global_epoch in tqdm(range(args.global_epoch)):
            batch_loss = []

            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                prediction  = global_model(data)
                loss        = loss_function(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            train_loss.append(sum(batch_loss)/len(batch_loss))

        test_accuracy = print_evaluation(args, global_model, test_dataset, true_test_label)

        end = time.time()

        learning_time = end - start
        print(f'\nTotal Run Time: {learning_time:.4f}')
        learning_time_sim.append(learning_time)

        model_path = 'model/' + str(args.terminal_number) + '_terminal_global_model_cl_' + args.propagation_loss + '.pth'
        torch.save(global_model, model_path)

        # Allocation
        if all_path_loss:
            test_accuracy_sim.append(test_accuracy)
            ts_n, ts_f, ts_nf, alloc_time = run_allocation_loop_all_path_loss(args, model_path)
            ts_normal_sim.append(sum(ts_n) / len(ts_n))
            ts_fspl_sim.append(sum(ts_f) / len(ts_f))
            ts_nfspl_sim.append(sum(ts_nf) / len(ts_nf))
            allocation_time_sim.append(sum(alloc_time) / len(alloc_time))
        else:
            satis_list, alloc_time = run_allocation_loop(args, model_path, 'hfl', device)
            satis_harmean_sim.append(sum(satis_list) / len(satis_list))
            allocation_time_sim.append(sum(alloc_time) / len(alloc_time))

    if all_path_loss:
        return ts_normal_sim, ts_fspl_sim, ts_nfspl_sim, test_accuracy_sim, learning_time_sim, allocation_time_sim
    else:
        return satis_harmean_sim, learning_time_sim, allocation_time_sim


# ══════════════════════════════════════════════════════════════════════════════
# Learning phase: Federated Learning
# ══════════════════════════════════════════════════════════════════════════════

def run_fl(args, device, logger, all_path_loss=False):
    print('\n======= Federated Learning =======')

    if all_path_loss:
        ts_normal_sim = []; ts_fspl_sim = []; ts_nfspl_sim = []
        test_accuracy_sim = []; learning_time_sim = []; allocation_time_sim = []
    else:
        satis_harmean_sim = []; learning_time_sim = []; allocation_time_sim = []

    for simulation in tqdm(range(args.simulation_number)):
        start = time.time()

        dataset_outputs, trained_kpca = load_supervised_dataset(args)
        train_dataset, test_dataset, true_test_label, train_application_number = dataset_outputs
        user_group = partition_dataset(args.dirichlet, train_dataset, train_application_number, args.min_required_size, args.max_required_size, args.terminal_number, args.alpha)

        input_number = len(train_dataset[0][0])
        global_model = build_global_model(args, input_number, device)

        # MOON local model pool
        if args.update_method == 'moon':
            previous_local_model_set = []
            local_model_pool = {idx: copy.deepcopy(global_model) for idx in range(args.terminal_number)}

        train_loss = []; train_accuracy = []
        best = copy.deepcopy(global_model); max_train_accuracy = 0.0

        for global_epoch in tqdm(range(args.global_epoch)):
            print(f'\n | Global Training Round : {global_epoch + 1} |\n')

            local_model_set = []; local_loss = []; sample_number = []

            user_number_max = max(int(args.user_fraction * args.terminal_number), 1)
            user_index = np.random.choice(range(args.terminal_number), user_number_max, replace=False)

            for index in user_index:
                m = args.update_method
                if m == 'fedavg' or m == 'feddf':
                    local_model = Local_Update(args, train_dataset, user_group[index], logger)
                    updated, loss = local_model.update_weight(copy.deepcopy(global_model))
                elif m == 'fedprox':
                    local_model = Local_Update_FedProx(args, train_dataset, user_group[index], logger)
                    updated, loss = local_model.update_weight(copy.deepcopy(global_model), copy.deepcopy(global_model))
                elif m == 'scaffold':
                    local_model = Local_Update_Scaffold(args, train_dataset, user_group[index], logger)
                    user_model = copy.deepcopy(global_model)
                    for op, np_ in zip(user_model.parameters(), global_model.parameters()):
                        op.data = np_.data.clone()
                    updated, loss = local_model.update_weight(user_model, copy.deepcopy(global_model))
                elif m == 'fednova':
                    local_model = Local_Update_FedNova(args, train_dataset, user_group[index], logger)
                    updated, loss = local_model.update_weight(copy.deepcopy(global_model), copy.deepcopy(global_model))
                elif m == 'moon':
                    local_model = Local_Update_MOON(args, train_dataset, user_group[index], logger)
                    previous_local_model = [prev[index] for prev in previous_local_model_set]
                    updated, loss = local_model.update_weight(copy.deepcopy(global_model), copy.deepcopy(global_model), previous_local_model)
                    local_model_pool[index] = updated

                local_model_set.append(copy.deepcopy(updated))
                local_loss.append(copy.deepcopy(loss))
                sample_number.append(int(0.8 * len(user_group[index])))

            # Aggregation
            m = args.update_method
            if m in ('fedavg', 'fedprox'):
                global_model = average_weight(local_model_set, sample_number, copy.deepcopy(global_model))
            elif m == 'scaffold':
                global_model = average_weight_scaffold(local_model_set, sample_number, copy.deepcopy(global_model), args)
            elif m == 'fednova':
                global_model = average_weight_fednova(local_model_set, sample_number, copy.deepcopy(global_model))
            elif m == 'feddf':
                global_model = average_weight(local_model_set, sample_number, copy.deepcopy(global_model))
                global_model = feddf_distillation(args, global_model, local_model_set, device, fitted_kpca = trained_kpca)
            elif m == 'moon':
                global_model = average_weight(local_model_set, sample_number, copy.deepcopy(global_model))
                if len(previous_local_model_set) < args.model_buffer_size:
                    snap = copy.deepcopy(local_model_pool)
                    for _, lm in snap.items():
                        lm.eval()
                        for p in lm.parameters():
                            p.requires_grad = False
                    previous_local_model_set.append(snap)
                elif args.previous_model_pool_option == 'FIFO':
                    snap = copy.deepcopy(local_model_pool)
                    for _, lm in snap.items():
                        lm.eval()
                        for p in lm.parameters():
                            p.requires_grad = False
                    for i in range(args.model_buffer_size - 2, -1, -1):
                        previous_local_model_set[i] = previous_local_model_set[i + 1]
                    previous_local_model_set[args.model_buffer_size - 1] = snap

            train_loss.append(sum(local_loss) / len(local_loss))

            # Per-terminal accuracy
            list_acc = []
            for index in range(args.terminal_number):
                lm = Local_Update_MOON(args, train_dataset, user_group[index], logger) if args.update_method == 'moon' else Local_Update(args, train_dataset, user_group[index], logger)
                acc, _ = lm.inference(global_model)
                list_acc.append(acc)
            train_accuracy.append(sum(list_acc) / len(list_acc))

            print(f'\nAverage Training Statistics after {global_epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}%\n'.format(100 * train_accuracy[-1]))

            if train_accuracy[-1] > max_train_accuracy:
                max_train_accuracy = train_accuracy[-1]
                best = global_model

        global_model = best if args.best else global_model
        test_accuracy = print_evaluation(args, global_model, test_dataset, true_test_label)

        end = time.time()
        learning_time = end - start
        print(f'\nTotal Run Time: {learning_time:.4f}')
        learning_time_sim.append(learning_time)

        model_path = (
            'model/' + str(args.terminal_number) + '_terminal_global_model_fl_' + args.update_method + '_' + args.propagation_loss + '.pth'
        )
        torch.save(global_model, model_path)

        # Allocation
        if all_path_loss:
            test_accuracy_sim.append(test_accuracy)
            ts_n, ts_f, ts_nf, alloc_time = run_allocation_loop_all_path_loss(args, model_path, trained_kpca)
            ts_normal_sim.append(sum(ts_n) / len(ts_n))
            ts_fspl_sim.append(sum(ts_f) / len(ts_f))
            ts_nfspl_sim.append(sum(ts_nf) / len(ts_nf))
            allocation_time_sim.append(sum(alloc_time) / len(alloc_time))
        else:
            satis_list, alloc_time = run_allocation_loop(args, model_path, 'fl', device, trained_kpca)
            satis_harmean_sim.append(sum(satis_list) / len(satis_list))
            allocation_time_sim.append(sum(alloc_time) / len(alloc_time))

    if all_path_loss:
        return ts_normal_sim, ts_fspl_sim, ts_nfspl_sim, test_accuracy_sim, learning_time_sim, allocation_time_sim
    else:
        return satis_harmean_sim, learning_time_sim, allocation_time_sim


# ══════════════════════════════════════════════════════════════════════════════
# Learning phase: Hierarchical Federated Learning
# ══════════════════════════════════════════════════════════════════════════════

def run_hfl(args, device, logger, all_path_loss=False):
    print('\n======= Hierarchical Federated Learning =======')

    if all_path_loss:
        ts_normal_sim = []; ts_fspl_sim = []; ts_nfspl_sim = []
        test_accuracy_sim = []; learning_time_sim = []; allocation_time_sim = []
    else:
        satis_harmean_sim = []; learning_time_sim = []; allocation_time_sim = []

    for simulation in tqdm(range(args.simulation_number)):
        start = time.time()

        dataset_outputs, trained_kpca = load_supervised_dataset(args)
        train_dataset, test_dataset, true_test_label, train_application_number = dataset_outputs
        user_group = partition_dataset(args.dirichlet, train_dataset, train_application_number, args.min_required_size, args.max_required_size, args.terminal_number, args.alpha)

        input_number = len(train_dataset[0][0])
        global_model = build_global_model(args, input_number, device)

        # Build HFL structure
        m = args.update_method
        if m in ('fedavg', 'fedprox'):
            structure = Structure(args, global_model, test_dataset)
        elif m == 'scaffold':
            structure = Structure_Scaffold(args, global_model, test_dataset)
        elif m == 'fednova':
            structure = Structure_FedNova(args, global_model, test_dataset)
        elif m == 'feddf':
            structure = Structure_FedDF(args, global_model, test_dataset)
        elif m == 'moon':
            structure = Structure_MOON(args, global_model, test_dataset)

        train_loss = []; train_accuracy = []
        best = copy.deepcopy(global_model); max_train_accuracy = 0.0

        for global_epoch in tqdm(range(args.global_epoch)):
            print(f'\n | Global Training Round : {global_epoch + 1} |\n')

            for mid_epoch in range(args.mid_epoch):
                local_model_set  = {}
                local_loss       = []
                user_server_index = {}

                user_number_max = max(int(args.user_fraction * args.terminal_number), 1)
                user_index = np.random.choice(range(args.terminal_number), user_number_max, replace=False)

                for index in user_index:
                    if m == 'fedavg' or m == 'feddf':
                        local_model = Local_Update(args, train_dataset, user_group[index], logger)
                    elif m == 'fedprox':
                        local_model = Local_Update_FedProx(args, train_dataset, user_group[index], logger)
                    elif m == 'scaffold':
                        local_model = Local_Update_Scaffold(args, train_dataset, user_group[index], logger)
                    elif m == 'fednova':
                        local_model = Local_Update_FedNova(args, train_dataset, user_group[index], logger)
                    elif m == 'moon':
                        local_model = Local_Update_MOON(args, train_dataset, user_group[index], logger)

                    if m == 'scaffold':
                        user_model, inter_server_model, server_index = structure.get_model(index, args.download)
                    elif m == 'moon':
                        prev_local, inter_server_model, server_index = structure.get_model(index, args.download)
                    else:
                        inter_server_model, server_index = structure.get_model(index, args.download)

                    if server_index in user_server_index:
                        user_server_index[server_index].append(index)
                    else:
                        user_server_index[server_index] = [index]

                    if m == 'fedavg':
                        updated, loss = local_model.update_weight(inter_server_model)
                    elif m == 'fedprox':
                        updated, loss = local_model.update_weight(copy.deepcopy(inter_server_model), inter_server_model)
                    elif m == 'scaffold':
                        updated, loss = local_model.update_weight(user_model, inter_server_model)
                        structure.system[-1][index] = copy.deepcopy(updated)
                    elif m == 'fednova':
                        updated, loss = local_model.update_weight(copy.deepcopy(inter_server_model), inter_server_model)
                    elif m == 'feddf':
                        updated, loss = local_model.update_weight(inter_server_model)
                    elif m == 'moon':
                        updated, loss = local_model.update_weight(copy.deepcopy(inter_server_model), inter_server_model, prev_local)

                    structure.system[-1][index] = updated

                    if server_index in local_model_set:
                        local_model_set[server_index].append(updated)
                    else:
                        local_model_set[server_index] = [updated]

                    local_loss.append(loss)
                
                if m == 'feddf':
                    structure.upload_weight(local_model_set, user_server_index, user_group, mid_epoch, fitted_kpca = trained_kpca)
                else:
                    structure.upload_weight(local_model_set, user_server_index, user_group, mid_epoch)
                train_loss.append(sum(local_loss) / len(local_loss))

            # Per-terminal accuracy
            list_acc = []
            for index in range(args.terminal_number):
                lm = Local_Update_MOON(args, train_dataset, user_group[index], logger) if m == 'moon' else Local_Update(args, train_dataset, user_group[index], logger)
                if m == 'scaffold':
                    user_model_inf, _, _ = structure.get_model(index, args.download)
                elif m == 'moon':
                    _, user_model_inf, _ = structure.get_model(index, args.download)
                else:
                    user_model_inf, _ = structure.get_model(index, args.download)
                acc, _ = lm.inference(user_model_inf)
                list_acc.append(acc)
            train_accuracy.append(sum(list_acc) / len(list_acc))

            print(f'\nAverage Training Statistics after {global_epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}%\n'.format(100 * train_accuracy[-1]))

            if train_accuracy[-1] > max_train_accuracy:
                max_train_accuracy = train_accuracy[-1]
                best = structure.system[1][0][1]

        global_model = best if args.best else structure.system[1][0][1]
        test_accuracy = print_evaluation(args, global_model, test_dataset, true_test_label)

        end = time.time()
        learning_time = end - start
        print(f'\nTotal Run Time: {learning_time:.4f}')
        learning_time_sim.append(learning_time)

        model_path = (
            'model/' + str(args.terminal_number) + '_terminal_global_model_hfl_' + args.update_method + '_' + args.propagation_loss + '.pth'
        )
        torch.save(global_model, model_path)

        # Allocation
        if all_path_loss:
            test_accuracy_sim.append(test_accuracy)
            ts_n, ts_f, ts_nf, alloc_time = run_allocation_loop_all_path_loss(args, model_path, trained_kpca)
            ts_normal_sim.append(sum(ts_n) / len(ts_n))
            ts_fspl_sim.append(sum(ts_f) / len(ts_f))
            ts_nfspl_sim.append(sum(ts_nf) / len(ts_nf))
            allocation_time_sim.append(sum(alloc_time) / len(alloc_time))
        else:
            satis_list, alloc_time = run_allocation_loop(args, model_path, 'hfl', device, trained_kpca)
            satis_harmean_sim.append(sum(satis_list) / len(satis_list))
            allocation_time_sim.append(sum(alloc_time) / len(alloc_time))

    if all_path_loss:
        return ts_normal_sim, ts_fspl_sim, ts_nfspl_sim, test_accuracy_sim, learning_time_sim, allocation_time_sim
    else:
        return satis_harmean_sim, learning_time_sim, allocation_time_sim


# ══════════════════════════════════════════════════════════════════════════════
# Learning phase: Greedy Algorithm
# ══════════════════════════════════════════════════════════════════════════════

def run_greedy(args):
    print('\n======= Greedy Algorithm =======')

    satis_harmean_sim  = []
    learning_time_sim  = [0] * args.simulation_number   # no learning phase
    allocation_time_sim = []

    for simulation in tqdm(range(args.simulation_number)):
        satis_list, alloc_time = run_allocation_loop(args, model_path=None, learning_method='greedy', device='cpu')
        satis_harmean_sim.append(sum(satis_list) / len(satis_list))
        allocation_time_sim.append(sum(alloc_time) / len(alloc_time))

    return satis_harmean_sim, learning_time_sim, allocation_time_sim


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    path_project = abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    if '--kpca' in sys.argv:
        # If run via CLI/run_all.py, find its positional text value
        kpca_idx = sys.argv.index('--kpca')
        if kpca_idx + 1 < len(sys.argv):
            cli_val = sys.argv[kpca_idx + 1].lower()
            args.kpca = cli_val in ['true', '1', 't', 'yes']
        print('c' * 30)
    else:
        # If run manually and --kpca wasn't in terminal command, force native bool
        args.kpca = bool(args.kpca)
    
    print(type(args.kpca))
        
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    lm           = args.learning_method
    all_path_loss = args.all_path_loss

    print(f'\n>>> Running method: {lm.upper()}' + (' (all path loss)' if all_path_loss else ''))

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if lm == 'cl':
        if all_path_loss:
            ts_n, ts_f, ts_nf, test_acc, learning_time, allocation_time = run_cl(args, device, logger, all_path_loss=True)

            print('normal', round(sum(ts_n)  / len(ts_n),  4))
            print('fspl',   round(sum(ts_f)  / len(ts_f),  4))
            print('nfspl',  round(sum(ts_nf) / len(ts_nf), 4))

            if args.simulation_number == 100:
                file = File_Save_Propagation_Loss(args)
                file.save(test_acc,       'test_accuracy',         args.propagation_loss)
                file.save(learning_time,  'learning_time',         args.propagation_loss)
                file.save(allocation_time,'allocation_time',       args.propagation_loss)
                file.save(ts_n,           'terminal_satisfaction', 'normal')
                file.save(ts_f,           'terminal_satisfaction', 'fspl')
                file.save(ts_nf,          'terminal_satisfaction', 'nfspl')
        else:
            satis, learning_time, allocation_time = run_cl(args, device, logger, all_path_loss=False)

            if args.simulation_number == 100:
                file = File_Save(args)
                file.save(learning_time,   'learning_time')
                file.save(allocation_time, 'allocation_time')
                file.save(satis,           'terminal_satisfaction')

    elif lm == 'fl':
        if all_path_loss:
            ts_n, ts_f, ts_nf, test_acc, learning_time, allocation_time = run_fl(args, device, logger, all_path_loss=True)

            print('normal', round(sum(ts_n)  / len(ts_n),  4))
            print('fspl',   round(sum(ts_f)  / len(ts_f),  4))
            print('nfspl',  round(sum(ts_nf) / len(ts_nf), 4))

            if args.simulation_number == 100:
                file = File_Save_Propagation_Loss(args)
                file.save(test_acc,      'test_accuracy',         args.propagation_loss)
                file.save(learning_time, 'learning_time',         args.propagation_loss)
                file.save(allocation_time,'allocation_time',      args.propagation_loss)
                file.save(ts_n,          'terminal_satisfaction', 'normal')
                file.save(ts_f,          'terminal_satisfaction', 'fspl')
                file.save(ts_nf,         'terminal_satisfaction', 'nfspl')
        else:
            satis, learning_time, allocation_time = run_fl(args, device, logger, all_path_loss=False)

            if args.simulation_number == 100:
                file = File_Save(args)
                file.save(learning_time,   'learning_time')
                file.save(allocation_time, 'allocation_time')
                file.save(satis,           'terminal_satisfaction')

    elif lm == 'hfl':
        if all_path_loss:
            ts_n, ts_f, ts_nf, test_acc, learning_time, allocation_time = run_hfl(args, device, logger, all_path_loss=True)

            print('normal', round(sum(ts_n)  / len(ts_n),  4))
            print('fspl',   round(sum(ts_f)  / len(ts_f),  4))
            print('nfspl',  round(sum(ts_nf) / len(ts_nf), 4))

            if args.simulation_number == 100:
                file = File_Save_Propagation_Loss(args)
                file.save(test_acc,       'test_accuracy',         args.propagation_loss)
                file.save(learning_time,  'learning_time',         args.propagation_loss)
                file.save(allocation_time,'allocation_time',       args.propagation_loss)
                file.save(ts_n,           'terminal_satisfaction', 'normal')
                file.save(ts_f,           'terminal_satisfaction', 'fspl')
                file.save(ts_nf,          'terminal_satisfaction', 'nfspl')
        else:
            satis, learning_time, allocation_time = run_hfl(args, device, logger, all_path_loss=False)

            if args.simulation_number == 100:
                file = File_Save(args)
                file.save(learning_time,   'learning_time')
                file.save(allocation_time, 'allocation_time')
                file.save(satis,           'terminal_satisfaction')

    elif lm == 'greedy':
        satis, learning_time, allocation_time = run_greedy(args)

        if args.simulation_number == 100:
            file = File_Save(args)
            file.save(learning_time,   'learning_time')
            file.save(allocation_time, 'allocation_time')
            file.save(satis,           'terminal_satisfaction')