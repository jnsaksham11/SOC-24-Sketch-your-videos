import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pickle

def load_data(file="data.pkl"):
    def vectorize(j):
        y = np.zeros((10,))
        y[j] = 1.0
        return y

    print("loading data...")
    with open(file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        training_inputs = [np.reshape(x, (784,)) for x in training_data[0]]
        training_outputs = [vectorize(y) for y in training_data[1]]
        training_data = list(zip(training_inputs, training_outputs))
        test_inputs = [np.reshape(x, (784,)) for x in test_data[0]]
        test_outputs = [vectorize(y) for y in test_data[1]]
        test_data = list(zip(test_inputs, test_outputs))
    print("loaded data...")
    return training_data, test_data

data_training, data_test = load_data()
X_train, Y_train = zip(*data_training)
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = zip(*data_test)
X_test, Y_test = np.array(X_test), np.array(Y_test)

# print(X_train[0].shape)

model = models.Sequential([
    layers.Dense(8, activation='relu', input_shape=(784,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=7, batch_size=40, validation_data=(X_test, Y_test))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

# print(model.predict(X_train[0].reshape(1,784)))
