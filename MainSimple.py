import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nhzworks.NeuralNetwork import NeuralNetwork
import sys
from Utils import Utils

#########################################################
# Set your hyperparameters here
##########################################################
iterations = 500 #100
learning_rate = 0.3
hidden_nodes = 11 #2
output_nodes = 1


def MSE(y, Y):
    return np.mean((y-Y)**2)


utils = Utils() 

N_i = utils.train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(utils.train_features.index, size=128)
    X, y = utils.train_features.ix[batch].values, utils.train_targets.ix[batch]['cnt']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(utils.train_features).T, utils.train_targets['cnt'].values)
    val_loss = MSE(network.run(utils.val_features).T, utils.val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()

fig, ax = plt.subplots(figsize=(8,4))

mean, std = utils.scaled_features['cnt']
predictions = network.run(utils.test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((utils.test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(utils.rides.ix[utils.test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

plt.show()