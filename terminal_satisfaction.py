from access_point import network_indicator_connected

class Terminal_Satisfaction():
    def __init__(self, args, access_point_set, terminal_set):
        self.args = args
        self.access_point_set = access_point_set
        self.terminal_set = terminal_set

    def terminal_satisfaction(self):
        satisfaction_sum_reciprocal = 0

        for terminal in self.terminal_set:
            _, satisfaction_reciprocal = self.each_terminal_satisfaction(terminal)
            satisfaction_sum_reciprocal += satisfaction_reciprocal
        
        satisfaction_harmonic_mean = self.args.terminal_number / round(satisfaction_sum_reciprocal, 6)
        return satisfaction_harmonic_mean

    def each_terminal_satisfaction(self, terminal):
        satisfaction = 0
        satisfaction_reciprocal = 0

        tp_connected = self.access_point_set[terminal.access_point_id].tp_connected
        rtt_connected = self.access_point_set[terminal.access_point_id].rtt_connected

        if terminal.network_indicator == 'tp':
            satisfaction = tp_connected / terminal.tp_need
            satisfaction_reciprocal = 1 / round(satisfaction, 6)

        else:
            satisfaction = terminal.rtt_need / rtt_connected
            satisfaction_reciprocal = 1 / round(satisfaction, 6)
        
        return satisfaction, satisfaction_reciprocal

class Terminal_Satisfaction_Propagation_Loss(Terminal_Satisfaction):
    def __init__(self, args, access_point_set, terminal_set, propagation_loss):
        super().__init__(args, access_point_set, terminal_set)
        self.propagation_loss = propagation_loss
    
    def terminal_satisfaction(self):
        satisfaction_sum_reciprocal = 0

        for i, terminal in enumerate(self.terminal_set):
            propagation_loss_rate_set = self.propagation_loss[i]
            access_point_id = propagation_loss_rate_set[terminal.access_point_id]
            propagation_loss_rate = access_point_id.rate
            _, satisfaction_reciprocal = self.each_terminal_satisfaction(terminal, propagation_loss_rate)
            satisfaction_sum_reciprocal += satisfaction_reciprocal
        
        satisfaction_harmonic_mean = self.args.terminal_number / round(satisfaction_sum_reciprocal, 6)
        return satisfaction_harmonic_mean

    def each_terminal_satisfaction(self, terminal, propagation_loss_rate):
        satisfaction = 0
        satisfaction_reciprocal = 0

        tp_connected = self.access_point_set[terminal.access_point_id].tp_connected * propagation_loss_rate
        rtt_connected = self.access_point_set[terminal.access_point_id].rtt_connected

        if terminal.network_indicator == 'tp':
            satisfaction = tp_connected / terminal.tp_need
            satisfaction_reciprocal = 1 / round(satisfaction, 6)

        else:
            satisfaction = terminal.rtt_need / rtt_connected
            satisfaction_reciprocal = 1 / round(satisfaction, 6)
        
        return satisfaction, satisfaction_reciprocal

class Terminal_Satisfaction_After_Allocation(Terminal_Satisfaction):
    def __init__(self, args, access_point_set, terminal_set, combination_set):
        super().__init__(args, access_point_set, terminal_set)
        self.combination_set = combination_set
    
    def terminal_satisfaction_after_allocation(self):
        # print(self.combination_set)
        for index, terminal in enumerate(self.terminal_set):
            terminal.access_point_id_set(self.combination_set[index])
        
        network_indicator_connected(self.args, self.access_point_set, self.terminal_set)

        return self.terminal_satisfaction()
    
class Terminal_Satisfaction_After_Allocation_Propagation_Loss(Terminal_Satisfaction_Propagation_Loss):
    def __init__(self, args, access_point_set, terminal_set, propagation_loss, combination_set):
        super().__init__(args, access_point_set, terminal_set, propagation_loss)
        self.combination_set = combination_set
    
    def terminal_satisfaction_after_allocation(self):
        for index, terminal in enumerate(self.terminal_set):
            terminal.access_point_id_set(self.combination_set[index])
        
        network_indicator_connected(self.args, self.access_point_set, self.terminal_set)

        return self.terminal_satisfaction()