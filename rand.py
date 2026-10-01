import math
import random

def access_point_random(args, terminal_set):
    for terminal in terminal_set:
        access_point_id_random = math.floor(random.random() * args.access_point_number)
        terminal.access_point_id_set(access_point_id_random)

def application_random(args, terminal_set):
    for terminal in terminal_set:
        application_id_random = math.floor(random.random() * args.application_number)
        terminal.application_information_set(application_id_random)