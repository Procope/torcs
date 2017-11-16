import csv
import numpy as np

from sklearn.utils import shuffle as parallel_shuffle
from sklearn.decomposition import PCA
from sklearn import preprocessing

from inputs import In
import pickle

def read_data(filepath, shuffle=True, pca_dims=7):
    """
    Read training data.
    Args:
        filepath: a csv file
        shuffle: whether to randomly shuffle the training data
        pca_dims: number of dimensions for dimensionality reduction; 0 means no PCA
    """
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        x = []
        y1 = []
        y2 = []
        y3 = []

        # skip the headers
        std_length = len(next(reader, None)) + 1

        for line in reader:
            if len(line) != std_length:
                continue

            y1.append(float(line[0]))
            y2.append(float(line[1]))
            y3.append(float(line[2]))

            x.append(list(map(float, line[3:])))

        x = np.array(x, dtype='float32')
        y1 = np.array(y1, dtype='float32')
        y2 = np.array(y2, dtype='float32')
        y3 = np.array(y3, dtype='float32')

        scaling = []
        for i in range(x.shape[1]):
            scaling.append((np.mean(x[:,i]), np.std(x[:,i])))

        with open('scaling.pickle', 'wb') as f:
            pickle.dump(scaling, f)
    
        x = preprocessing.scale(x)

        if pca_dims > 0:
            pca = PCA(n_components = pca_dims).fit(x)
            with open('pca.pickle', 'wb') as f:
                pickle.dump(pca.components_, f)
            x = pca.transform(x)

    if shuffle:
        return parallel_shuffle(x, y1, y2, y3)
    else:
        return x, y1, y2, y3


def read_data_in_sequences(filepath, sequence_length, shuffle=True, pca_dims=7):
    """
    Read training data so to use sequences of state features as inputs.
    Args:
        filepath: a csv file
        sequence_length: the number of timesteps of which a sequence consists
        shuffle: whether to randomly shuffle the training data
        pca_dims: number of dimensions; 0 means no PCA
    """
    pca_dims_ = pca_dims
    x, y1, y2, y3 = read_data(filepath, shuffle=False, pca_dims=pca_dims_)

    x_seq = []

    n_features = x.shape[1]
    for i, features in enumerate(x, start=1):
        seq_features = np.zeros((sequence_length * n_features,))
        if i < sequence_length:
            rand_indices = list(map(int, x.shape[0] * np.random.random_sample((sequence_length-i,))))

            seq_features[:(sequence_length-i)*n_features] = np.concatenate([x[j] for j in rand_indices])

            # For the first inputs, this portion of the feature vector is empty
            try:
                seq_features[(sequence_length-i)*n_features:(sequence_length-1)*n_features] = x_seq[i-2][sequence_length:]
            except IndexError:
                pass
            except ValueError:
                pass

            seq_features[(sequence_length-1)*n_features:] = x[i-1]
        else:
            past = np.array(x_seq[-1])[n_features:]
            seq_features = np.concatenate([past, x[i-1]])

        x_seq.append(seq_features)

    x_seq = np.array(x_seq, dtype='float32')

    if shuffle:
        return parallel_shuffle(x_seq, y1, y2, y3)
    else:
        return x_seq, y1, y2, y3


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


def generate_batches(x, y, batch_size=128):
    assert(x.shape[0] == y.shape[0])

    n_batches = x.shape[0] // batch_size

    # Discarding the last batch for now, for simplicity.
    x_ = np.zeros(
        shape=(
            n_batches,
            batch_size,
            x.shape[1]),
        dtype=np.float32)
    y_ = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            x_[batch, idx] = x[(batch * batch_size) + idx]
            y_[batch, idx] = y[(batch * batch_size) + idx]

    return (x_, y_)


x, y1, y2, y3 = read_data_in_sequences('train_data/all-tracks.csv', 3)

