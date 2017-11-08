import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD

from sklearn import preprocessing

from reader import read_data, split_data


# the data, shuffled and split between train, validation, and test sets
x, y1, y2, y3 = read_data('train_data/alpine-1.csv', shuffle=True)
x = preprocessing.scale(x)

train_set, valid_set, test_set = split_data(x, y1, y2, y3, 0.8, 0.1)

x_train, y1_train, y2_train, y3_train = train_set
x_valid, y1_valid, y2_valid, y3_valid = valid_set
x_test, y1_test, y2_test, y3_test = test_set

batch_size = len(x_train)

epochs = 50

model_accel = Sequential()
model_accel.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model_accel.add(Dropout(0.1))
model_accel.add(Dense(1, activation='sigmoid'))

model_accel.summary()

model_accel.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.005, decay=1e-3),
              metrics=['accuracy'])

history = model_accel.fit(x_train, y1_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y1_test))

score = model_accel.evaluate(x_test, y1_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model_brake = Sequential()
model_brake.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model_brake.add(Dropout(0.1))
model_brake.add(Dense(1, activation='sigmoid'))

model_brake.summary()

model_brake.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.005, decay=1e-3),
              metrics=['accuracy'])

history = model_brake.fit(x_train, y2_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y2_test))

score = model_brake.evaluate(x_test, y2_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model_steer = Sequential()
model_steer.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model_steer.add(Dropout(0.1))
model_steer.add(Dense(1, activation='linear'))

model_steer.summary()

model_steer.compile(loss='mean_squared_error',
              optimizer=SGD())

history = model_steer.fit(x_train, y3_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y3_test))

score = model_steer.evaluate(x_test, y3_test, verbose=0)
print('Test loss:', score[0])

