from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
import math
import pickle
from forward_func import forward
import os
import logging

logger = logging.getLogger(__name__)

class MyDriver(Driver):
    def __init__(self, network_file='node_evals.pickle', network=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if network:
            self.network = [(node, bias, response, links) for (node, act_func, agg_func, bias, response, links) in network.node_evals] 
        else:   
            with open(network_file, 'rb') as net_in:
                self.network = pickle.load(net_in)
        self.max_angle = 35
        self.prev_command = Command()
        self.prev_state = None
        self.T_out = 0
        self.ticks = 0
        self.default = True
        self.start_distance = 0
        self.racer = True
        self.prev_dist = 200
        # [os.remove(file) for file in os.listdir() if file.startswith('pos_')]        

    def drive(self, carstate: State) -> Command:
        if self.ticks == 0:
            self.start_distance = carstate.distance_from_start

            with open('pos_' + str(carstate.race_position), 'wb') as posf:
                pickle.dump(carstate.race_position, posf)
                self.pos_file = 'pos_' + str(carstate.race_position)
        
        # if self.default: print(self.default)
        if self.ticks == 100:
            for p in range(40):
                if p == carstate.race_position:
                    continue
                try:
                    with open('pos_' + str(p), 'rb') as other_pos:
                        self.other_file = 'pos_' + str(p)
                        self.other_pos = pickle.load(other_pos)
                        self.racer = carstate.race_position < self.other_pos
                        # print(self.other_pos)
                    break
                except FileNotFoundError as ex:
                    continue

            # print(self.racer)


        if hasattr(self, 'other_file') and self.ticks > 100 and self.ticks % 50 == 0:
            with open(self.pos_file, 'wb') as posf:
                pickle.dump(carstate.race_position, posf)

            with open(self.other_file, 'rb') as otherf:
                try:
                    self.other_pos = pickle.load(otherf)
                    self.racer = carstate.race_position < self.other_pos
                except EOFError as err:
                    print(err)

            # print(self.racer, carstate.race_position, self.other_pos)


        if abs(carstate.distance_from_center) >= 1: 
            self.T_out += 1
        self.ticks += 1

        if not self.default and (abs(carstate.angle) >= self.max_angle or abs(carstate.distance_from_center) >= .8):
            self.default = True
            # print('default on')
        elif self.default and abs(carstate.angle) < self.max_angle-15 and abs(carstate.distance_from_center) < .8:
            self.default = False
            # print('default off')

        command = Command()
        if self.default:
            try:
                self.steer(carstate, 0.0, command)
                command.accelerator = 0.4
            except Exception as ex:
                self.default = False
                # print('default off')
                command = self.compute_command(carstate)    
        else:
            command = self.compute_command(carstate)

        ratio = 2.1
        if carstate.speed > 20 and max(carstate.front_edge_sensors) > 20 and max(carstate.front_edge_sensors)/carstate.speed < ratio:
            command.brake = (1/(ratio-1))*(carstate.speed / max(carstate.front_edge_sensors))
        
        self.shift(carstate, command)

        if not self.default and not self.racer and self.other_pos - carstate.race_position < 5:

            if min(carstate.opponents[7:10]) < 10 and self.prev_dist > min(carstate.opponents[7:10]):
                # print(self.ticks, carstate.race_position, 'bump')
                command.steering = -0.1
                self.prev_dist = min(carstate.opponents[7:10])

            elif min(carstate.opponents[-10:-7]) < 10 and self.prev_dist > min(carstate.opponents[-10:-7]):
                command.steering = 0.1
                self.prev_dist = min(carstate.opponents[-10:-7])
                # print(self.ticks, carstate.race_position, 'bump')

        if self.data_logger:
            self.data_logger.log(carstate, command)        

        self.prev_command = command
        self.prev_state = carstate
        return command

    def compute_command(self, carstate: State):
        accel, steer = forward(self.network, self.to_input(carstate))
        return self.to_command(accel, steer)

    def shift(self, carstate: State, command: Command):
        if carstate.rpm > 9000:
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

        input.append(carstate.opponents[8])
        input.append(carstate.opponents[15])
        input.append(carstate.opponents[18])
        input.append(carstate.opponents[20])
        input.append(carstate.opponents[29])
        input.append(carstate.opponents[-1])

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
        if not self.prev_state:
            return (-self.T_out*0.02 - 90, 
                self.T_out, 
                0, 
                0,
                self.ticks, 
                0,
                0, 
                0,
                0,
                self.start_distance,)

        if self.prev_state.distance_from_start == self.prev_state.distance_from_start:
            if self.prev_state.last_lap_time > 0:
                distance = track_length 
            elif self.prev_state.distance_raced <= (track_length - self.start_distance): 
                distance = self.prev_state.distance_raced
            else:
                distance = self.prev_state.distance_from_start
        else:
            distance = 0

        if self.ticks:
            speed_avg = distance/self.ticks
        else:
            speed_avg = 0        

        
        fitness = distance - self.T_out*0.02- 10*(self.prev_state.race_position - 1) #- 0.01*self.prev_state.damage 
        if self.prev_state.last_lap_time:
            fitness -= self.prev_state.last_lap_time
        else:
            fitness -= self.prev_state.current_lap_time

        return (fitness, 
                self.T_out, 
                self.prev_state.damage, 
                distance, 
                self.ticks, 
                speed_avg, 
                self.prev_state.last_lap_time, 
                self.prev_state.current_lap_time,
                self.prev_state.race_position,
                self.start_distance,)

    def on_shutdown(self):
        if hasattr(self, 'pos_file'):
            os.remove(self.pos_file)
        super().on_shutdown()
