import pickle 
import sys, traceback
import numpy as np

if __name__ == "__main__" and len(sys.argv) == 2:
    with open(sys.argv[1], 'rb') as fin, open('data.csv', 'w') as fout:
        while True:
            try:                
                state, command = pickle.load(fin)
                speed = np.linalg.norm([state.speed_x, state.speed_y, state.speed_z])
                line = str(command.accelerator)
                line += ',' + str(command.brake)
                line += ',' + str(command.steering)
                line += ',' + str(speed)
                line += ',' + str(state.distance_from_center)
                line += ',' + str(state.angle)
                line += ','.join([str(x) for x in state.distances_from_edge])
                fout.write(line + '\n')
            except EOFError:
                break
        