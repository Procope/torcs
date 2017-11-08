import csv
import numpy as np
from sklearn.utils import shuffle as parallel_shuffle
from inputs import In

def read_data(filepath, shuffle=True):
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        x = []
        y1 = []
        y2 = []
        y3 = []

        # skip the headers
        next(reader, None)

        for line in reader:
            y1.append(float(line[0]))
            y2.append(float(line[1]))
            y3.append(float(line[2]))

            x.append(list(map(float, line[3:])))

        x = np.array(x, dtype='float32')
        y1 = np.array(y1, dtype='float32')
        y2 = np.array(y2, dtype='float32')
        y3 = np.array(y3, dtype='float32')

    if shuffle:
        return parallel_shuffle(x, y1, y2, y3)
    else:
        return x, y1, y2, y3


def split_data(x, y1, y2, y3, train, valid):
    N = x.shape[0]
    train_size = int(N * train)
    valid_size = int(N * valid)

    def split(data, train_size, valid_size):
        data_train = data[:train_size]
        data_valid = data[train_size:train_size+valid_size]
        data_test = data[train_size+valid_size:]
        return data_train, data_valid, data_test

    x_train, x_valid, x_test = split(x, train_size, valid_size)
    y1_train, y1_valid, y1_test = split(y1, train_size, valid_size)
    y2_train, y2_valid, y2_test = split(y2, train_size, valid_size)
    y3_train, y3_valid, y3_test = split(y3, train_size, valid_size)

    train_set = (x_train, y1_train, y2_train, y3_train)
    valid_set = (x_valid, y1_valid, y2_valid, y3_valid)
    test_set = (x_test, y1_test, y2_test, y3_test)

    return train_set, valid_set, test_set

x, y1, y2, y3 = read_data('train_data/alpine-1.csv', shuffle=True)
train_set, valid_set, test_set = split_data(x, y1, y2, y3, 0.8, 0.1)
