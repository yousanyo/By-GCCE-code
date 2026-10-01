import numpy as np
from access_point import network_indicator_connected
from terminal_satisfaction import Terminal_Satisfaction
from terminal_satisfaction import Terminal_Satisfaction_Propagation_Loss
import math
import sys

class Hungarian():
    def __init__(self, args, access_point_set, terminal_set):
        self.args = args
        self.access_point_set = access_point_set
        self.terminal_set = terminal_set
        self.access_point_number = len(self.access_point_set)
        self.terminal_number = len(self.terminal_set)

    def hungarian(self):
        cost_matrix = np.zeros((self.terminal_number, self.terminal_number))
        hungarian_result_all = []

        access_point_terminal_combination = self.access_point_terminal_combination()
        for i in range(len(access_point_terminal_combination)):
            for j, terminal in enumerate(self.terminal_set):
                terminal.access_point_id_set(access_point_terminal_combination[i][j])
                
            network_indicator_connected(self.access_point_set, self.terminal_set)

            for j in range(self.terminal_number):
                access_point_connection = access_point_terminal_combination[i][j]
                for k, terminal in enumerate(self.terminal_set):
                    terminal.access_point_id_set(access_point_connection)
                    terminal_satisfaction = Terminal_Satisfaction(self.args, self.access_point_set, self.terminal_set)
                    satisfaction, _ = terminal_satisfaction.each_terminal_satisfaction(terminal)
                    cost_matrix[j][k] = round(satisfaction, 6)
                
            hungarian_result = self.hungarian_calculation(cost_matrix, access_point_terminal_combination[i])
            hungarian_result_all.append(hungarian_result)
        
        index_max = max(enumerate(hungarian_result_all), key = lambda x: x[1].harmonic_mean)[0]
        access_point_terminal_id_list = hungarian_result_all[index_max].access_point_terminal_matching

        '''
        terminal_satisfaction_max = hungarian_result_all[index_max].harmonic_mean
        print('Max of terminal satisfaction harmonic mean', terminal_satisfaction_max, 'Combination No.', index_max)
        print(f'Access point id after allocation: {access_point_terminal_id_list}')
        '''

        return access_point_terminal_id_list
    
    def access_point_terminal_combination(self):
        n = self.access_point_number + self.terminal_number - 1
        r = self.access_point_number - 1
        data = np.empty(n, dtype = float)
        division = [[]]

        access_point_count = np.zeros(self.access_point_number)
        access_point_terminal_combination = []

        for index in range(self.access_point_number):
            access_point_count[index] = 0
        for index in range(n):
            data[index] = index
        
        division = self.combination(data, r)
        
        for index in range(len(division)):
            division_no = 0
            access_point_no = 0
            j = 0
            wk_access_point_terminal_combination_tmp = []
            wk_access_point_terminal_combination = []
            
            while j < n:
                if j == division[index][division_no]:
                    wk_access_point_terminal_combination_tmp.append(-1)
                    access_point_no += 1
                    if division_no == 0:
                        division_no += 1
                else:
                    wk_access_point_terminal_combination_tmp.append(access_point_no)
                j += 1
            
            j = 0
            while j < n:
                if wk_access_point_terminal_combination_tmp[j] != -1:
                    wk_access_point_terminal_combination.append(wk_access_point_terminal_combination_tmp[j])
                j += 1
            
            access_point_terminal_combination.append(wk_access_point_terminal_combination.copy())
        
        return access_point_terminal_combination
    
    def combination(self, data, r):
        n = len(data)
        combination_set = []

        if r == 1:
            i = 0
            while i < n:
                combination_set.append([data[i]])
                i += 1
        
        else:
            i = 0
            count = n - r + 1
            while i < count:
                ts = self.combination(data[(i + 1):], r - 1)
                for t in ts:
                    t.insert(0, data[i])
                    combination_set.append(t)
                i += 1

        return combination_set

    def hungarian_calculation(self, cost_matrix, access_point_terminal_combination):
        n = self.terminal_number
        b = np.empty((n, n))
        fix_digit = math.pow(10, 6)
        nomi_solution = []
        wk_solution_value = np.empty(3)
        wk_solution_station = np.full(n, -1)
        nomi_solution_station = []

        for i in range(n):
            for j in range(n):
                b[i][j] = math.floor(cost_matrix[i][j] * fix_digit)
        
        fx = np.full(n, -sys.maxsize)
        fy = np.zeros(n)
        x = np.full(n, -1)
        y = np.full(n, -1)
        
        for i in range(n):
            for j in range(n):
                fx[i] = max(fx[i], b[i][j])
            
        i = 0
        while i < n:
            t = []
            s = []

            j = 0
            while j < n:
                t.append(-1)
                s.append(i)
                j += 1
            
            s.append(i)

            p = 0
            q = 0
            while p <= q and x[i] < 0:
                k = s[p]
                j = 0

                while j < n and x[i] < 0:
                    if fx[k] + fy[j] == b[k][j] and t[j] < 0:
                        q += 1
                        s[q] = y[j]
                        t[j] = k

                        if s[q] < 0:
                            p = j

                            while p >= 0:
                                y[j] = t[j]
                                k = t[j]
                                p = x[k]
                                x[k] = j
                                j = p
                    j += 1
                p += 1
        
            if x[i] < 0:
                d = sys.maxsize

                k = 0
                while (k <= q):

                    j = 0
                    while(j < n):
                        
                        if t[j] < 0:
                            d = min(d, fx[s[k]] + fy[j] - b[s[k]][j])
                        j += 1
                    k += 1

                j = 0
                while j < n:
                    if t[j] < 0:
                        fy[j] += 0

                    else:
                        fy[j] += d
                    j += 1

                k = 0
                while k <= q:
                    fx[s[k]] -= d
                    k += 1

            else:
                i += 1
    
        nomi_solution.append(x)

        min_max = 100
        wk_solution_value[0] = 0
        wk_solution_value[2] = 0
        for i in range(n):
            satisfaction = cost_matrix[i][x[i]]
            wk_solution_value[0] += satisfaction
            wk_solution_value[2] += 1 / round(satisfaction, 6)

            if satisfaction < min_max:
                min_max = satisfaction
        
        wk_solution_value[1] = min_max

        for i, new_index in enumerate(x):
            wk_solution_station[new_index] = access_point_terminal_combination[i]

        nomi_solution_station.append(wk_solution_station)

        result = self.Hungarian_Result()
        result.harmonic_mean = n / round(wk_solution_value[2], 6)
        result.access_point_terminal_matching = nomi_solution_station[0]
        
        return result

    class Hungarian_Result():
        harmonic_mean: float
        access_point_terminal_matching: list

class Hungarian_Propagation_Loss(Hungarian):
    def __init__(self, args, access_point_set, terminal_set, propagation_loss):
        super().__init__(args, access_point_set, terminal_set)
        self.propagation_loss = propagation_loss
    
    def hungarian(self):
        cost_matrix = np.zeros((self.terminal_number, self.terminal_number))
        hungarian_result_all = []
        access_point_terminal_combination = self.access_point_terminal_combination()
        terminal_satisfaction = Terminal_Satisfaction_Propagation_Loss(self.args, self.access_point_set, self.terminal_set, self.propagation_loss)

        for i in range(len(access_point_terminal_combination)):
            for j, terminal in enumerate(self.terminal_set):
                terminal.access_point_id_set(access_point_terminal_combination[i][j])
                
            network_indicator_connected(self.args, self.access_point_set, self.terminal_set)

            for j in range(self.terminal_number):
                access_point_connection = access_point_terminal_combination[i][j]
                for k, terminal in enumerate(self.terminal_set):
                    terminal.access_point_id_set(access_point_connection)

                    propagation_loss_rate_set = self.propagation_loss[k]
                    access_point_id = propagation_loss_rate_set[terminal.access_point_id]
                    propagation_loss_rate = access_point_id.rate
                    satisfaction, _ = terminal_satisfaction.each_terminal_satisfaction(terminal, propagation_loss_rate)
                    cost_matrix[j][k] = round(satisfaction, 6)
                
            hungarian_result = self.hungarian_calculation(cost_matrix, access_point_terminal_combination[i])
            hungarian_result_all.append(hungarian_result)
        
        index_max = max(enumerate(hungarian_result_all), key = lambda x: x[1].harmonic_mean)[0]
        access_point_terminal_id_list = hungarian_result_all[index_max].access_point_terminal_matching

        '''
        terminal_satisfaction_max = hungarian_result_all[index_max].harmonic_mean
        print('Max of terminal satisfaction harmonic mean', terminal_satisfaction_max, 'Combination No.', index_max)
        print(f'Access point id after allocation: {access_point_terminal_id_list}')
        '''

        return access_point_terminal_id_list