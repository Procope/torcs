from pytocl.driver import Driver
from pytocl.car import State, Command
from keras.models import load_model
import keras.backend as K
import numpy as np
import time
from pprint import pprint
import pickle

class MyDriver(Driver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accel = load_model('model_accel.h5')
        self.model_brake = load_model('model_brake.h5')
        self.model_steer = load_model('model_steer.h5')
        self.cnt = 0
        self.scaling = [(133.864,54.9646),(-0.00863886,0.50894),(0.00188822,0.105689),(6.5347,3.37256),(6.63965,3.43778),(6.95333,3.60714),(7.53029,3.91342),(8.49204,4.5185),(10.0121,5.27447),(12.6311,6.85156),(18.0477,10.4204),(39.4379,29.5398),(82.1273,51.8961),(41.5571,25.8683),(19.7458,14.6121),(13.9527,10.8658),(10.9412,8.62758),(9.22592,7.23537),(8.20196,6.45132),(7.57763,5.96612),(7.23648,5.69632),(7.12393,5.60116)]
        self.means = [mean_std[0] for mean_std in self.scaling]
        self.scales = [mean_std[1] for mean_std in self.scaling]
        with open('pca.pickle', 'rb') as f:
            self.pca =pickle.load(f)
        self.sequential = np.zeros((1,30))


    def drive(self, carstate: State) -> Command:
        return super().drive(carstate)

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
        input = self.to_input(carstate)
        accel = self.model_accel.predict([self.sequential])[0][0]
        brake = self.model_brake.predict([self.sequential])[0][0]
        steer = self.model_steer.predict([self.sequential])[0][0]

        return self.to_command(accel, brake, steer)

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

        self.sequential = np.concatenate((self.sequential, np.array([input])), axis=1)[0][-30:].reshape((1,30))

    def to_command(self, accelerate, brake, steer):
        command = Command()
        
        command.accelerator = accelerate
        command.brake = brake
        command.steering = steer

        return command
