from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

from numpy import genfromtxt


trainsize = 3300

inputs = 264
genres = 10

X = genfromtxt('train_data.csv', delimiter=',')

Y_0 = genfromtxt('train_labels.csv', delimiter=',')

print(X.shape)

Y = np.zeros((Y_0.shape[0], genres))

X += np.random.rand(X.shape[0], X.shape[1])*0.0000000001
X = (X - np.mean(X, axis=0) ) / np.std(X, axis=0)

for i in range(Y_0.shape[0]):
	index = int(Y_0[i]) - 1
	Y[i][index] = 1.0

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
model.add(Dense(30, input_dim=inputs, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(15, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(genres, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X0 = X[:trainsize]
Y0 = Y[:trainsize]

X1 = X[trainsize:]
Y1 = Y[trainsize:]


model.fit(X0, Y0, epochs=100, batch_size=10)

# evaluate the model
scores = model.evaluate(X1, Y1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
