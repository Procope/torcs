from pytocl.driver import Driver
from pytocl.car import State, Command
from keras.models import load_model
import numpy as np
class MyDriver(Driver):
    def __init__(*args, **kwargs):
    	super().__init__(*args, **kwargs)
    	self.model = load_model('model.h5')

    def drive(self, carstate: State) -> Command:
	    command = self.compute_command(carstate)
	    self.shift(carstate, command)

	    if self.data_logger:
	        self.data_logger.log(carstate, command)

	    return command

    def shift(carstate: State, command: Command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

	def compute_command(carstate: State):
		input = self.to_input(carstate: State)
		output = self.model.predict(input)
		return self.to_command(output)

	def to_input(carstate: State):
		speed = np.linalg.norm([carstate.speed_x, carstate.speed_y, carstate.speed_z])
		return np.array([speed, TRACKPOSITION, carstate.angle]
			.extend(carstate.distances_from_edge))

	def to_command(output):
		command = Command()
		
		command.accelerator = output[0]
        command.brake = output[1]
        command.steering = output[2]

        return command