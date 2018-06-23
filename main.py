import numpy as np
np.random.seed(100)

# import tensorboard as tb
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
# from keras.utils import to_categorical
import h5py


# SAMPLE_NUM = 2224
# INPUT_DIM = 92


def normalize_input(inputs):
    # normalise wifi record strength
    wr_inputs = inputs[:, 600:]
    zero_index = np.where(wr_inputs == 0)
    wr_inputs[zero_index] = -100

    max = np.max(wr_inputs)
    min = np.min(wr_inputs)

    wr_inputs = (wr_inputs - min)/(max-min)

    return wr_inputs


def reduce_grid_index(y):
        y = np.array(y, dtype='int')
        y = y.ravel()

        # get distinct grid index
        exist_grid_list = sorted(set(y))
        num_classes = len(exist_grid_list)
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=np.float32)

        for label, i in zip(y, range(n)):
            categorical[i, np.where(exist_grid_list == label)] = 1

        return exist_grid_list, categorical


def data_shuffle_split(name):

    dataset = h5py.File(name, "r")

    for name, i in zip(dataset.keys(), range(len(dataset.keys()))):
        print(np.shape(dataset[name]))
        if i:
            X_Y = np.vstack((X_Y, dataset[name]))
        else:
            X_Y = dataset[name]
        print("--------")
    dataset.close()
    np.random.shuffle(X_Y)
    SAMPLE_NUM = np.shape(X_Y)[0]
    INPUT_DIM = np.shape(X_Y)[1]- 3 - 2*300

    X = X_Y[:, 3:]
    X = normalize_input(X)
    grid_list, Y = reduce_grid_index(X_Y[:, 2])

    chunk_size = int(0.2*SAMPLE_NUM)
    Y_test = Y[0:chunk_size, :]
    X_test = X[0:chunk_size, :]

    # Y_val = Y[chunk_size: 2*chunk_size, :]
    # X_val = X[chunk_size: 2*chunk_size, :]

    Y_train = Y[chunk_size:, :]
    X_train = X[chunk_size:, :]

    # return X_train, Y_train, X_val, Y_val, X_test, Y_test, grid_list, SAMPLE_NUM, INPUT_DIM
    return X_train, Y_train, X_test, Y_test, grid_list, SAMPLE_NUM, INPUT_DIM


def main():

    # Read data from file into memory
    f_name = "./background_results/out_in_overall.h5"
    # x_train, y_train, x_val, y_val, x_test, y_test, grid_list, SAMPLE_NUM, INPUT_DIM = data_shuffle_split(f_name)
    x_train, y_train, x_test, y_test, grid_list, SAMPLE_NUM, INPUT_DIM = data_shuffle_split(f_name)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=INPUT_DIM))
    # model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(16, activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(len(grid_list), activation="softmax"))

    # sgd = SGD(lr=0.05, decay=1e-3, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=12)

    score = model.evaluate(x_test, y_test, batch_size=12)

    print(score)


if __name__ == "__main__":
    main()