"""
This project takes in the MNIST training dataset and does stuff
"""

import numpy as np
import pandas as pd
# import torch;

train_df = pd.read_csv('digit-recognizer/train.csv')

train_labels = train_df['label']
train_df.drop(['label'],inplace=True, axis=1)

train_df = train_df.to_numpy()
train_input = np.empty(shape=(1,28,28))

# print(np.shape(train_df)) # --> (42000, 784)

two_d = np.reshape(train_df[0],[28,28])
train_input[0] = two_d

for ind in range(1,np.shape(train_df)[0]):
	if ind == 10: break
	two_d = np.reshape(train_df[ind],[28,28])
	np.append(train_input, two_d)

print(train_input[0])
print(np.shape(train_input))