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
    def __init__(self, network_file='network_best.pickle', network=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if network:
            self.network =  network
        else:   
            with open(network_file, 'rb') as net_in:
                self.network = pickle.load(net_in)
        self.max_angle = 30
        self.prev_command = Command()
        self.prev_state = None
        self.T_out = 0
        self.ticks = 0
        self.default = False

    def drive(self, carstate: State) -> Command:
        if abs(carstate.distance_from_center) >= 1: 
            self.T_out += 1
        self.ticks += 1

        if not self.default and (abs(carstate.angle) >= self.max_angle or abs(carstate.distance_from_center) >= 1):
            self.default = True
        elif self.default and abs(carstate.angle) < self.max_angle-15 and abs(carstate.distance_from_center) < 1:
            self.default = False

        command = Command()
        if self.default:
            self.steer(carstate, 0.0, command)
            command.accelerator = 0.2
        elif max(carstate.distances_from_edge[7:12:2]) > 200:
            command.accelerator = 1
        else:
            command = self.compute_command(carstate)

        self.shift(carstate, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)        

        self.prev_command = command
        self.prev_state = carstate
        return command

    def compute_command(self, carstate: State):
        accel, steer = self.network.activate(self.to_input(carstate))
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
        input.append(max(carstate.distances_from_edge[7:12:2]))

        return input

    def to_command(self, accelerate, steer):
        command = Command()
        
        if accelerate >= 0:
            command.accelerator = accelerate
        else:
            command.brake = abs(accelerate)
        
        command.steering = (self.prev_command.steering + steer)/2

        return command

    def eval(self, track_length):
        # check for NaN
        if self.prev_state.distance_from_start == self.prev_state.distance_from_start and self.prev_state.current_lap_time:
            if self.prev_state.last_lap_time > 0:
                distance = track_length
            else: 
                distance = self.prev_state.distance_from_start
        else:
            distance = 0

        speed_avg = distance/self.ticks
        
        print('T_out:     ', self.T_out)
        print('distance:  ', distance)
        print('ticks:     ', self.ticks)
        print('speed:     ', speed_avg)
        print('time:      ', self.prev_state.last_lap_time)
        
        if self.prev_state.last_lap_time:
            return distance - self.prev_state.last_lap_time
        else:
            return distance - self.prev_state.current_lap_time