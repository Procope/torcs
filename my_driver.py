from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
import numpy as np
from scipy.special import expit
import time
import math
from pprint import pprint
import pickle
import neat
import logging
_logger = logging.getLogger(__name__)

class MyDriver(Driver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        with open('../torcs/network_winner.pickle', 'rb') as net_in:
            self.model = pickle.load(net_in)
        self.max_angle = 30
        self.prev_command = Command()
        self.T_out = 0
        self.distance = 0
        self.ticks = 0
        self.time = time.time()
        self.total_time = 0

    def drive(self, carstate: State) -> Command:
        # return super().drive(carstate)
        time1 = time.time()
        
        if abs(carstate.distance_from_center) >= 1: 
            self.T_out += 1

        self.distance = carstate.distance_raced
        self.ticks += 1

        command = Command()
        if abs(carstate.angle) >= self.max_angle or abs(carstate.distance_from_center) >= 1:
            self.steer(carstate, 0.0, command)
            command.accelerator = 0.2
        elif carstate.distances_from_edge[9] > 100:
            command.accelerator = 1
        else:
            command = self.compute_command(carstate)

        self.shift(carstate, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)        

        time2 = time.time()
        # print(time2-self.time)
        self.total_time += time2 - self.time
        self.time = time2
        print(self.distance)
        # print(self.total_time)
        return command

    def compute_command(self, carstate: State):
        accel, steer = self.model.activate(self.to_input(carstate))
        return self.to_command(accel, steer)

    def shift(self, carstate: State, command: Command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500 and carstate.gear > 1:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
    
    def to_input(self, carstate: State):
        speed = np.linalg.norm([carstate.speed_x, carstate.speed_y, carstate.speed_z])
        input = [speed]
        input.append(carstate.distances_from_edge[0])
        input.append(carstate.distances_from_edge[2])
        input.append(carstate.distances_from_edge[4])
        input.append(carstate.distances_from_edge[-1])
        input.append(carstate.distances_from_edge[-3])
        input.append(carstate.distances_from_edge[-5])
        input.append(sum(carstate.distances_from_edge[7:12:2]))

        return input

    def to_command(self, accelerate, steer):
        command = Command()
        
        if accelerate >= 0:
            command.accelerator = accelerate
        else:
            command.brake = abs(accelerate)
        
        command.steering = (self.prev_command.steering + steer)/2

        return command

    def on_shutdown(self):
        super().on_shutdown()
        with open('eval.pickle', 'wb') as eval_out:
            pickle.dump((self.T_out, self.distance, self.ticks), eval_out)
