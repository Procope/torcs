import numpy as np
np.random.seed(0)

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop, SGD

from sklearn import preprocessing

from reader import read_data, read_data_in_sequences, split_data, generate_batches


# the data, shuffled and split between train, validation, and test sets
x, y1, y2, y3 = read_data_in_sequences('train_data/all-tracks.csv', 3, shuffle=True, pca_dims=7)

train_set, valid_set, test_set = split_data(x, y1, y2, y3, 0.8, 0.1)

x_train, y1_train, y2_train, y3_train = train_set
x_valid, y1_valid, y2_valid, y3_valid = valid_set
x_test, y1_test, y2_test, y3_test = test_set

batch_size = 128
epochs = 70

# model_accel = Sequential()
# model_accel.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
# model_accel.add(Dropout(0.1))
# model_accel.add(Dense(1, activation='sigmoid'))
#
# model_accel.summary()
#
# model_accel.compile(loss='binary_crossentropy',
#               optimizer=RMSprop(lr=0.005, decay=1e-3),
#               metrics=['accuracy'])
#
# history = model_accel.fit(x_train, y1_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_valid, y1_valid))
#
# model_accel.save('model_accel.h5')
# score = model_accel.evaluate(x_test, y1_test, verbose=0)
# print("ACCELERATION\n")
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
#
# model_brake = Sequential()
# model_brake.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
# model_brake.add(Dropout(0.1))
# model_brake.add(Dense(1, activation='sigmoid'))
#
# model_brake.summary()
#
# model_brake.compile(loss='binary_crossentropy',
#               optimizer=RMSprop(lr=0.005, decay=1e-3),
#               metrics=['accuracy'])
#
# history = model_brake.fit(x_train, y2_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_valid, y2_valid))
#
# model_brake.save('model_brake.h5')
# score = model_brake.evaluate(x_test, y2_test, verbose=0)
# print("\n\nBREAK\n")
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

def mean_distance(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true-y_pred, 2)))

model_steer = Sequential()
model_steer.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model_steer.add(Dropout(0.1))
model_steer.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model_steer.add(Dropout(0.1))
model_steer.add(Dense(1, activation='tanh'))

model_steer.summary()

model_steer.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.1),
              metrics=[mean_distance])

history = model_steer.fit(x_train, y3_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_valid, y3_valid))

model_steer.save('model_steer.h5')

score = model_steer.evaluate(x_test, y3_test, verbose=0)
# preds = model_steer.predict(x_test, batch_size=batch_size)

# for i in range(5):
#   print(y3_test[i], preds[i])

print("\n\nSTEERING")
print('Test loss:', score)
