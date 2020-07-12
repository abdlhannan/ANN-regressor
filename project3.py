import numpy as np
import pandas as pd
# part 2
# Matplotlib for visualizing graphs
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
def param_init():
    weight_1 = np.random.normal(scale=10)
    bias_1 = np.random.normal(scale=10)
    return weight_1, bias_1
def normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)
def shuffle(x):
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    return x[index, :]
def loss_func(weight_1, bias_1, train):
    output = train[:,0] * weight_1 + bias_1
    return (1/2) * np.mean(np.square(output - train[:,1]))
file = open("train1", "r")
train = []
for x in file:
    temp = x.split('\t')
    #print(x)
    #print(temp[1].split('\n'))
    train.append([float(temp[0]), float(temp[1].split('\n')[0])])
train = np.array(train)    
file = open("test1", "r")
test = []
for x in file:
    temp = x.split('\t')
    #print(temp)
    test.append([float(temp[0]), float(temp[1].split('\n')[0])])
test = np.array(test)
LEARNING_RATE = 0.001
EPOCHS = 200
weight_1, bias_1 = param_init()
training_mse, testing_mse = [], []
losses = []
train = normalize(train)
test = normalize(test)
for epoch in range(EPOCHS):
    train = shuffle(train)
    for stoch_train in train:
        #FORWARD PROPOGATION
        output = stoch_train[0] * weight_1 + bias_1
        #BACKWARD PROPOGATION
        d_weight_1 = (output - stoch_train[1]) * stoch_train[0]
        d_bias_1 = output - stoch_train[1]
        #UPDATE PARAMETERS
        weight_1 = weight_1 - LEARNING_RATE * d_weight_1
        bias_1 = bias_1 - LEARNING_RATE * d_bias_1
    training_mse.append(loss_func(weight_1, bias_1, train))
    testing_mse.append(loss_func(weight_1, bias_1, test))
    if epoch%10==0:
        print('EPOCH: ', epoch, 'TRAIN LOSS: ', training_mse[epoch], 'TEST LOSS ',testing_mse[epoch])




# b)


def plot(data, weight, bias, error, training):
    x_line = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 10)
    y_pred = weight * x_line + bias
    fig = plt.figure(figsize=(8 * 2, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.scatter(data[:,0], data[:,1], alpha=0.8)
    ax1.plot(x_line, y_pred, linewidth=2, markersize=12, color='red', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Data point and prediction line')
    
    ax2.plot(error)
    if training:
        ax2.set_title('TRAINING MSE ERROR')
    else:    
        ax2.set_title('TESTING MSE ERROR')
    
        
    plt.show()
plot(train, weight_1, bias_1, training_mse, True)
plot(test, weight_1, bias_1, testing_mse, False)



#c)

def param_init(input_layer_size, hidden_layer_size, output_layer_size):
    weight_1 = np.random.randn(hidden_layer_size, input_layer_size) * 0.05
    bias_1 = np.zeros((hidden_layer_size, 1))
    
    weight_2 = np.random.randn(output_layer_size, hidden_layer_size) * 0.05
    bias_2 = np.zeros((output_layer_size, 1))
    
    return weight_1, bias_1, weight_2, bias_2
def normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)
def shuffle(x):
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    return x[index, :]
def loss_func(weight_1, bias_1, weight_2, bias_2, train):
    
    weighted_sum1 = np.dot(np.reshape(train[:, 0], (train.shape[0],1)),weight_1.T) + bias_1.T
    activation1 = 1/(1 + np.exp(-weighted_sum1))
    
    weighted_sum2 = np.dot(activation1, weight_2.T) + bias_2.T
    output = weighted_sum2
    
    return (1/2) * np.mean(np.square(output - train[:,1]))
def plot_2(data, weight_1, bias_1,weight_2, bias_2, error, training, hiddenunits):
    x_line = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 40)
    x_line = np.reshape(x_line, (x_line.shape[0],1))
    
    weighted_sum1 = np.dot(x_line, weight_1.T) + bias_1.T
    activation1 = 1/(1 + np.exp(-weighted_sum1))
    #print(activation1.shape)
    weighted_sum2 = np.dot(activation1, weight_2.T) + bias_2.T
    y_pred = weighted_sum2
    
    
    #y_pred = weight * x_line + bias
    fig = plt.figure(figsize=(8 * 2, 6))
    
    plt.plot(error)
    if training:
        plt.title('TRAINING MSE ERROR. Hidden Units: ' + str(hiddenunits))
    else:    
        plt.title('TESTING MSE ERROR. Hidden Units: ' + str(hiddenunits))
    
        
    plt.show()
def plot_2_pred(data, weight_1, bias_1,weight_2, bias_2, error, training):
    x_line = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 40)
    x_line = np.reshape(x_line, (x_line.shape[0],1))
    
    weighted_sum1 = np.dot(x_line, weight_1.T) + bias_1.T
    activation1 = 1/(1 + np.exp(-weighted_sum1))
    #print(activation1.shape)
    weighted_sum2 = np.dot(activation1, weight_2.T) + bias_2.T
    y_pred = weighted_sum2
    
    
    #y_pred = weight * x_line + bias
    fig = plt.figure(figsize=(8 * 2, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.scatter(data[:,0], data[:,1], alpha=0.8)
    ax1.plot(x_line, y_pred, linewidth=2, markersize=12, color='red', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Data point and prediction line')
    
    
def hiddenlayer_ann_regressor(epochs, lr, hiddenunits, optimalparams):
    file = open("train1", "r")
    train = []
    for x in file:
        temp = x.split('\t')
        #print(x)
        #print(temp[1].split('\n'))
        train.append([float(temp[0]), float(temp[1].split('\n')[0])])
    train = np.array(train)    
    file = open("test1", "r")
    test = []
    for x in file:
        temp = x.split('\t')
        #print(temp)
        test.append([float(temp[0]), float(temp[1].split('\n')[0])])
    test = np.array(test)   
    HIDDEN_LAYER_SIZE = hiddenunits
    LEARNING_RATE = lr
    EPOCHS = epochs
    weight_1, bias_1, weight_2, bias_2 = param_init(1, HIDDEN_LAYER_SIZE, 1)
    training_mse, testing_mse = [], []

    train = normalize(train)
    test = normalize(test)
    for epoch in range(EPOCHS):
        train = shuffle(train)
        for stoch_train in train:
            weighted_sum1 = np.dot(stoch_train[0], weight_1.T) + bias_1.T
            activation1 = 1/(1 + np.exp(-weighted_sum1))
            #print(activation1.shape)
            weighted_sum2 = np.dot(activation1, weight_2.T) + bias_2.T
            output = weighted_sum2
            #print(output)
            d_weighted_sum2 = output - stoch_train[1]
            d_weight2 = np.dot(d_weighted_sum2, activation1)
            d_bias2 = d_weighted_sum2

            d_weighted_sum1 = np.multiply(np.dot(d_weighted_sum2,weight_2), activation1*(1 - activation1))
            d_weight1 = np.dot(d_weighted_sum1.T, stoch_train[0])
            d_bias1 = d_weighted_sum1.T
            #print(d_weight2.shape)
            weight_2 = weight_2 - LEARNING_RATE * d_weight2
            bias_2 = bias_2 - LEARNING_RATE * d_bias2
            weight_1 = weight_1 - LEARNING_RATE * d_weight1
            bias_1 = bias_1 - LEARNING_RATE * d_bias1
            #print(weight_2.shape)
        training_mse.append(loss_func(weight_1, bias_1, weight_2, bias_2, train))
        testing_mse.append(loss_func(weight_1, bias_1, weight_2, bias_2, test))
    
    

    plot_2(train, weight_1, bias_1,weight_2,bias_2, training_mse, True, hiddenunits)
    plot_2(test, weight_1, bias_1,weight_2,bias_2, testing_mse, False, hiddenunits)        
    optimalparams[str(hiddenunits)] = [weight_1, weight_2, bias_1, bias_2, training_mse, testing_mse, LEARNING_RATE, EPOCHS, train.shape[0], test.shape[0]]
    return optimalparams
optimalparams = {}        
    
        
        

# graphs for part c
# Each Hidden Unit has seperate Training loss and Testing loss

optimalparams = hiddenlayer_ann_regressor(100, 0.0001, 2, optimalparams)    

optimalparams = hiddenlayer_ann_regressor(100, 0.0001, 4, optimalparams) 

optimalparams = hiddenlayer_ann_regressor(15, 0.0004, 8, optimalparams) 

optimalparams = hiddenlayer_ann_regressor(15, 0.0004, 16, optimalparams) 

optimalparams = hiddenlayer_ann_regressor(15, 0.0004, 32, optimalparams) 

training_error = []
testing_error = []
trainingerror_std = []
testingerror_std = [] 
for hidden_layer in optimalparams.keys():
    training_error.append(np.sum(optimalparams[hidden_layer][4])/optimalparams[hidden_layer][8])
    testing_error.append(np.sum(optimalparams[hidden_layer][5])/optimalparams[hidden_layer][9])
    trainingerror_std.append(np.std(optimalparams[hidden_layer][4]))
    testingerror_std.append(np.std(optimalparams[hidden_layer][5]))
df = pd.DataFrame({'hidden_layers':[2, 4, 8, 16, 32], 'Average Training Error':training_error,
                  'Training Error Standard Deviation':trainingerror_std, 'Average Testing Error':testing_error,
                  'Testing Error Standard Deviation':testingerror_std})


# Table for c part

print(df)



