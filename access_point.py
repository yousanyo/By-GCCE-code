import math
import copy
import numpy as np

class Access_Point_Structure:
    def __init__(self):
        self.access_point_id = 0
        self.tp_connected = 0.0
        self.rtt_connected = 0.0
        self.terminal_number = 0
        self.position = [0]
    
    def access_point_id_set(self, access_point_id):
        self.access_point_id = access_point_id
    
    def terminal_number_set(self, terminal_number):
        self.terminal_number = terminal_number

    def tp_set(self, tp):
        self.tp_connected = tp
    
    def rtt_set(self, rtt):
        self.rtt_connected = rtt
    
    def position_set(self):
        x = 30 * (self.access_point_id + 1)
        self.position.insert(0, x)

def create_access_point(access_point_number):
    access_point_set = []
    for index in range(access_point_number):
        access_point_structure = Access_Point_Structure()
        access_point_structure.access_point_id_set(index)
        access_point_structure.position_set()
        access_point_set.append(access_point_structure)
    return access_point_set
    
def network_indicator_connected(args, access_point_set, terminal_set):
    initial_rtt = [20.0, 28.0, 32.0]
    rtt_connected = copy.deepcopy(initial_rtt)
    initial_tp = [65500 * 2 * 8 / initial_rtt[0] / 1024, 65500 * 2 * 8 / initial_rtt[1] / 1024, 65500 * 2 * 8 / initial_rtt[2] / 1024]
    tp_connected = copy.deepcopy(initial_tp)

    erlang_mu = [50, 50, 20]
    erlang_n = [10, 5, 3]
    respo_mu = [1 / initial_rtt[0] * 1000, 1 / initial_rtt[1] * 1000, 1 / initial_rtt[2] * 1000]

    def erlang_b(lambda_val, mu, n):
        rho = lambda_val / mu
        b = (rho ** n / math.factorial(n)) / sum((rho ** k) / math.factorial(k) for k in range(n + 1))
        return b

    def response_time_mm1(lambda_val, mu):
        if lambda_val >= mu - 1.9:
            lambda_val = mu - 1 + (lambda_val-mu) * 0.01
        return 1 / (mu - lambda_val)
    
    access_point_terminal_number = access_point_terminal_sum(args, access_point_set, terminal_set)
    for index, access_point in enumerate(access_point_set):
        erlang = erlang_b(access_point_terminal_number[index] * 100, erlang_mu[index], erlang_n[index])
        tp_connected[index] = initial_tp[index] * (1 - erlang)

        response_time = response_time_mm1(access_point_terminal_number[index], respo_mu[index])
        rtt_connected[index] = response_time * 1000

        if tp_connected[index] <= 0:
            tp_connected[index] = 0.01

        access_point.tp_set(tp_connected[index])
        access_point.rtt_set(rtt_connected[index])

def access_point_terminal_sum(args, access_point_set, terminal_set):
    access_point_terminal_number = np.zeros(args.access_point_number)

    for terminal in terminal_set:
        access_point_terminal_number[terminal.access_point_id] += 1
    
    for index, access_point in enumerate(access_point_set):
        access_point.terminal_number_set(access_point_terminal_number[index])
    
    return access_point_terminal_number