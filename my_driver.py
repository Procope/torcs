from pytocl.driver import Driver
from pytocl.car import State, Command
from keras.models import load_model
import keras.backend as K
import numpy as np
from scipy.special import expit
import time
from pprint import pprint
import pickle
from lstm import mixed_loss, accuracy_test, mean_distance

class MyDriver(Driver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        with open('scaling.pickle', 'rb') as f:
            scaling = pickle.load(f)
            self.means = [mean_std[0] for mean_std in scaling]
            self.scales = [mean_std[1] for mean_std in scaling]

        with open('pca.pickle', 'rb') as f:
            self.pca = pickle.load(f)

        custom_objects = {'mixed_loss': mixed_loss,
                          'accuracy_test': accuracy_test,
                          'mean_distance': mean_distance
                          }
        self.model = load_model('lstm.h5', custom_objects=custom_objects)

        self.input_shape = tuple(self.model.layers[0].input_shape[1:])
        self.input = np.zeros(self.input_shape)
        self.cnt = 0


    def drive(self, carstate: State) -> Command:
        # return super().drive(carstate)

        time1 = time.time()
        command = self.compute_command(carstate)
        self.shift(carstate, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        time2 = time.time()
        # print('function took %0.3f ms' % ((time2-time1)*1000.0))
        print(command)
        return command

    def compute_command(self, carstate: State):
        self.to_input(carstate)
        accel, brake, steer = self.model.predict(self.input.reshape((1,7,7)))[0]
        return self.to_command(expit(accel), expit(brake), steer)

    def shift(self, carstate: State, command: Command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500 and carstate.gear > 2:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
    
    def to_input(self, carstate: State):
        speed = np.linalg.norm([carstate.speed_x, carstate.speed_y, carstate.speed_z])
        input = [speed, carstate.distance_from_center, carstate.angle]
        input.extend(carstate.distances_from_edge)

        input = np.divide(np.subtract(input, self.means), self.scales)
        input = np.matmul(self.pca, input)
        self.input = np.append(self.input[1:], input.reshape(1,7), axis=0)

    def to_command(self, accelerate, brake, steer):
        command = Command()
        
        command.accelerator = accelerate
        command.brake = brake
        command.steering = steer

        return command
