import random
import numpy as np

def load_dataset(dataset_path, num_input, num_classes):
    """
    Loads the dataset into array of sequences.
    """
    if dataset_path is None:
        exit(20) #TODO error
    ds_file = open(dataset_path, 'r')
    if ds_file is None:
        exit(21) #TODO error

    lines = ds_file.readlines()
    
    dataset_x = []
    dataset_y = []

    last_tag = -1
    sequence = []
    count = 0
    for line in lines:
        line = line[:-1]
        line_data = line.split(" ")
        if line_data[0]=="tag":
            continue

        c = int(line_data[-1])
        tag = int(line_data[0])
        if last_tag!=tag:
            count = count+1
            if count!=1:
                dataset_x.append(sequence)
                dataset_y.append(y)
            y = [0 if i!=c else 1 for i in range(num_classes)]
            sequence = []
            last_tag= tag
        sequence.extend(line_data[1:num_input+1])
    
    dataset_x.append(sequence)
    dataset_y.append(y)
    
    ds_file.close()

    return dataset_x, dataset_y


def divide_data(train_x, train_y, test_x, test_y, num_input, num_classes, timesteps, overlapdim=0):
    """
    Divide dataset into overlapping temporal windows of size (timesteps, num_input).
    Windows from the same sequence might be put in different sets.
    """    
    step = num_input*(timesteps-overlapdim)
    n_windows = int(len(train_x[0])/step)

    while (n_windows-1)*step+(num_input*timesteps) > len(train_x[0]):
        n_windows = n_windows-1

    new_train_x = [np.array(train_x[i][j*step:(j*step)+(timesteps*num_input)]) for j in range(n_windows) for i in range(len(train_x))]
    x_shape = (n_windows*len(train_x), timesteps, num_input)
    new_train_x = np.reshape(np.array(new_train_x), x_shape)
    y_shape = (len(train_x)*n_windows, num_classes)
    new_train_y = np.reshape(np.array(train_y*n_windows), y_shape)
    
    sequence_length = len(test_x[0])
    x_shape = (len(test_x), sequence_length)
    test_x = np.reshape(np.array(test_x), x_shape)
    y_shape = (len(test_y), num_classes)
    test_y = np.reshape(np.array(test_y), y_shape)
    test_y = np.argmax(test_y,axis=1)

    return new_train_x, new_train_y, test_x, test_y
