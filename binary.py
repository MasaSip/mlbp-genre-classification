from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

from numpy import genfromtxt


trainsize = 3300

inputs = 264
genres = 1

X = genfromtxt('train_data.csv', delimiter=',')

Y_0 = genfromtxt('train_labels.csv', delimiter=',')

print(X.shape)

Y = np.zeros(Y_0.shape[0])

#X += np.random.rand(X.shape[0], X.shape[1])*0.000000000001
X = (X- np.min(X, axis=0)) / (np.max(X, axis=0)-np.min(X, axis=0)+0.0000001)

print(X)

for i in range(Y_0.shape[0]):
	index = int(Y_0[i]) - 1
	if index == 0:
		Y[i] = 1.0

print(X)
print("Y before", Y)
print(Y.shape)

ind = np.arange(X.shape[0])
print(ind)
np.random.shuffle(ind)
print(X.shape)
X = X[ind]
Y = Y[ind]
X.shape
print(ind)
print("Y after", Y)

# create model
model = Sequential()
model.add(Dense(30, input_dim=inputs, activation='relu', bias_initializer='ones'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu', bias_initializer='ones'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='relu', bias_initializer='ones'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='relu', bias_initializer='ones'))
model.add(Dense(genres, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X0 = X[:trainsize]
Y0 = Y[:trainsize]

X1 = X[trainsize:]
Y1 = Y[trainsize:]


model.fit(X0, Y0, epochs=2500, batch_size=3300)

# evaluate the model
scores = model.evaluate(X1, Y1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
