import json
import random

with open('application.json', 'r') as f:
    application_information = json.load(f)

def create_terminal(args):
    terminal_set = []
    for index in range(args.terminal_number):
        terminal = Terminal_Structure()
        terminal.terminal_id_set(index)
        if args.learning_method == 'hungarian':
            if args.propagation_loss == 'fspl':
                terminal.fspl_position_set()
            elif args.propagation_loss == 'nfspl':
                terminal.nfspl_position_set()
        else:
            terminal.nfspl_position_set()
        terminal_set.append(terminal)
    return terminal_set
        
class Terminal_Structure:
    def __init__(self):
        self.terminal_id = 0
        self.access_point_id = 0
        self.position = []
        self.application_id = 0
        self.network_indicator = ''
        self.tp_need = 0.0
        self.rtt_need = 0.0
    
    def terminal_id_set(self, terminal_id):
        self.id = terminal_id

    def access_point_id_set(self, access_point_id):
        self.access_point_id = access_point_id
    
    def fspl_position_set(self):
        x_ranges = [(0, 29), (32, 59), (62, 89), (92, 121)]
        x_selected_range = random.choice(x_ranges)
        x_float = random.uniform(*x_selected_range)
        y_ranges = [(-30, -1), (2, 31)]
        y_selected_range = random.choice(y_ranges)
        y_float = random.uniform(*y_selected_range)
        self.position = [x_float, y_float]
    
    def nfspl_position_set(self):
        x_ranges = [(0, 29), (32, 59), (62, 89), (92, 121)]
        x_selected_range = random.choice(x_ranges)
        x_float = random.uniform(*x_selected_range)
        y_ranges = [(-30, -16), (-13, -1), (2, 14), (17, 31)]
        y_selected_range = random.choice(y_ranges)
        y_float = random.uniform(*y_selected_range)
        self.position = [x_float, y_float]
    
    def application_information_set(self, application_id):
        self.application_id = application_id
        self.network_indicator = application_information[self.application_id]["indicator"]
        self.tp_need = application_information[self.application_id]["tp_need"]
        self.rtt_need = application_information[self.application_id]["rtt_need"]

    