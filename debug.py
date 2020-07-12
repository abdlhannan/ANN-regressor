import numpy as np


def param_init(input_layer_size, hidden_layer_size, output_layer_size):
    weight_1 = np.random.randn(hidden_layer_size, input_layer_size) * 0.01
    bias_1 = np.zeros((hidden_layer_size, 1))

    weight_2 = np.random.randn(output_layer_size, hidden_layer_size) * 0.01
    bias_2 = np.zeros((output_layer_size, 1))

    return weight_1, bias_1, weight_2, bias_2


def normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)


def shuffle(x):
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    return x[index, :]


def loss_func(weight_1, bias_1, weight_2, bias_2, train):
    weighted_sum1 = np.reshape(train[:, 0], (train.shape[0], 1)) @ weight_1.T + bias_1.T
    activation1 = 1 / (1 + np.exp(-weighted_sum1))

    weighted_sum2 = activation1 @ weight_2.T + bias_2.T
    output = weighted_sum2

    return (1 / 2) * np.mean(np.square(output - train[:, 1]))


file = open("train1", "r")
train = []
for x in file:
    temp = x.split('\t')
    # print(x)
    # print(temp[1].split('\n'))
    train.append([float(temp[0]), float(temp[1].split('\n')[0])])
train = np.array(train)
file = open("test1", "r")
test = []
for x in file:
    temp = x.split('\t')
    # print(temp)
    test.append([float(temp[0]), float(temp[1].split('\n')[0])])
test = np.array(test)
HIDDEN_LAYER_SIZE = 4
LEARNING_RATE = 0.01
EPOCHS = 500
weight_1, bias_1, weight_2, bias_2 = param_init(1, HIDDEN_LAYER_SIZE, 1)
training_mse, testing_mse = [], []
losses = []
train = normalize(train)
test = normalize(test)
for epoch in range(EPOCHS):
    train = shuffle(train)
    for stoch_train in train:
        weighted_sum1 = stoch_train[0] * weight_1.T  # + bias_1.T
        activation1 = 1 / (1 + np.exp(-weighted_sum1))
        # print(activation1.shape)
        weighted_sum2 = activation1 @ weight_2.T  # + bias_2.T
        output = weighted_sum2
        # print(output)
        d_weighted_sum2 = output - stoch_train[1]
        d_weight2 = d_weighted_sum2.T * activation1
        d_bias2 = d_weighted_sum2

        d_weighted_sum1 = np.multiply(d_weighted_sum2 * weight_2, activation1 * (1 - activation1))
        d_weight1 = d_weighted_sum1.T * stoch_train[0]
        d_bias1 = d_weighted_sum1.T
        # print(d_weight2.shape)
        weight_2 = weight_2 - LEARNING_RATE * d_weight2
        # bias_2 = bias_2 - LEARNING_RATE * d_bias2
        weight_1 = weight_1 - LEARNING_RATE * d_weight1
        # bias_1 = bias_1 - LEARNING_RATE * d_bias1
        # print(weight_2.shape)
    print(loss_func(weight_1, bias_1, weight_2, bias_2, train))





