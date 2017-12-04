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
        self.max_angle = 35
        self.prev_command = Command()
        self.prev_state = None
        self.T_out = 0
        self.ticks = 0
        self.default = False
        self.start_distance = 0

    def drive(self, carstate: State) -> Command:
        if self.ticks == 0:
            self.start_distance = carstate.distance_from_start

        if abs(carstate.distance_from_center) >= 1: 
            self.T_out += 1
        self.ticks += 1

        if not self.default and (abs(carstate.angle) >= self.max_angle or abs(carstate.distance_from_center) >= 1):
            self.default = True
            print('default on')
        elif self.default and abs(carstate.angle) < self.max_angle-15 and abs(carstate.distance_from_center) < 1:
            self.default = False
            print('default off')

        command = Command()
        if self.default:
            try:
                self.steer(carstate, 0.0, command)
                command.accelerator = 0.2
            except Exception as ex:
                self.default = False
                print('default off')
                command = self.compute_command(carstate)    
        else:
            command = self.compute_command(carstate)

        if carstate.speed > 20 and max(carstate.front_edge_sensors) > 40 and max(carstate.front_edge_sensors) < 80 and max(carstate.front_edge_sensors)/carstate.speed < 2.0 and not command.brake:
            command.brake = 0.6*(carstate.speed / max(carstate.front_edge_sensors))
        
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
        speed = carstate.speed
        input = [speed]
        input.append(carstate.distances_from_edge[0])
        input.append(carstate.distances_from_edge[2])
        input.append(carstate.distances_from_edge[4])
        input.append(carstate.distances_from_edge[-1])
        input.append(carstate.distances_from_edge[-3])
        input.append(carstate.distances_from_edge[-5])
        input.append(max(carstate.distances_from_edge[7:12:2]))

        # input.append(carstate.opponents[8])
        # input.append(carstate.opponents[15])
        # input.append(carstate.opponents[18])
        # input.append(carstate.opponents[20])
        # input.append(carstate.opponents[29])
        # input.append(carstate.opponents[-1])

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
        if self.prev_state.distance_from_start == self.prev_state.distance_from_start:
            if self.prev_state.last_lap_time > 0:
                distance = track_length 
            else: 
                distance = min(self.prev_state.distance_from_start, self.prev_state.distance_raced)
        else:
            distance = 0

        speed_avg = distance/self.ticks
        
        print('T_out:     ', self.T_out)
        print('distance:  ', distance)
        print('ticks:     ', self.ticks)
        print('speed:     ', speed_avg)
        print('prev time: ', self.prev_state.last_lap_time)
        print('cur time:  ', self.prev_state.current_lap_time)
        print('position:  ', self.prev_state.race_position)
        print('start_distance: ', self.start_distance)
        
        fitness = distance - self.prev_state.damage - self.T_out*0.02
        if self.prev_state.last_lap_time:
            fitness -= self.prev_state.last_lap_time
        else:
            fitness -= self.prev_state.current_lap_time
        return fitness - 10*(self.prev_state.race_position - 1)

    def on_shutdown(self):
        super().on_shutdown()
        self.eval(2843.1)
