from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

from numpy import genfromtxt


def normalize(M):	
	return (M - np.mean(M, axis=0) ) / (np.std(M, axis=0) + np.mean(M, axis=0)*0.0000001)

trainsize = 3800
inputs = 264
genres = 10

X = genfromtxt('train_data.csv', delimiter=',')
X = normalize(X)
print(X.shape)
inputs = len(X[0])

Y_0 = genfromtxt('train_labels.csv', delimiter=',')

print(X.shape)

Y = np.zeros((Y_0.shape[0], genres))

numbers = np.arange(genres)

for i in range(Y_0.shape[0]):
	index = int(Y_0[i]) - 1
	numbers[index] += 1
	Y[i][index] = 1.0

np.random.seed(1)
ind = np.arange(X.shape[0])
np.random.shuffle(ind)
X = X[ind]
Y = Y[ind]
X.shape

# create model
model = Sequential()
model.add(Dense(inputs // 3, input_dim=inputs, activation='softplus', bias_initializer='ones'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='elu', bias_initializer='ones'))
model.add(Dropout(0.1))
model.add(Dense(14, activation='softplus'))
model.add(Dense(6, activation='elu', bias_initializer='ones'))
model.add(Dense(genres, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split to train and verification data
X0 = X[:trainsize]
Y0 = Y[:trainsize]

X1 = X[trainsize:]
Y1 = Y[trainsize:]

# Fit model
model.fit(X0, Y0, epochs=1000, batch_size=(trainsize // 2), verbose=1, validation_data=(X1, Y1))

scores = model.evaluate(X1, Y1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def print_outputs():
	Z = genfromtxt('test_data.csv', delimiter=',')
	Z = normalize(Z)
	print(Z.shape)

	logloss = model.predict(Z)
	print("logloss", logloss.shape, logloss)
	indices = np.repeat(np.arange(logloss.shape[0]) + 1, 1)
	gens = np.argmax(logloss, axis=1) + 1
	
	logloss = np.c_[indices, logloss] + 0.001
	#print("MIN PROBS", np.min(logloss))
	np.savetxt("logloss_foo.csv", logloss, delimiter=",", fmt="%i,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f")	

	print(gens.shape)
	print(indices.shape)
	gens = np.stack((indices,gens)).T
	gens = gens.astype(int)
	print("genres", gens.shape, gens)
	np.savetxt("genres_foo.csv", gens, delimiter=",", fmt="%i")

print_outputs()
