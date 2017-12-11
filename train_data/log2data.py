import sys
from pytocl.analysis import DataLogReader
import numpy as np

if __name__ == "__main__" and len(sys.argv) >= 2:
    state_attributes = ['speed_x', 'speed_y', 'speed_z', 'distance_from_center', 'angle', 'distances_from_edge']
    command_attributes = ['accelerator', 'brake', 'steering']
    reader = DataLogReader(sys.argv[1], state_attributes, command_attributes)
    
    filename_out = sys.argv[2] if len(sys.argv) > 2 else 'data.csv'
    with open(filename_out, 'w') as fout:
        header = 'ACCELERATION,BRAKE,STEERING,SPEED,TRACK_POSITION,ANGLE_TO_TRACK_AXIS,'
        header += ','.join(['TRACK_EDGE_' + str(i) for i in range(18)])
        fout.write(header + '\n')
        
        for log in reader.array:
            line = ','.join([str(x) for x in log[-3:]])
            line += ',' + str(np.linalg.norm([log[1:4]]))
            line += ',' + ','.join([str(x) for x in log[4:-3]])
            fout.write(line + '\n')
        