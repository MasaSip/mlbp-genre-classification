from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

from numpy import genfromtxt


def normalize(M):	
	return (M - np.mean(M, axis=0) ) / (np.std(M, axis=0) + 0.001* abs(np.mean(M, axis=0)))


trainsize = 3300

inputs = 50
#inputs = 264
genres = 10

X = genfromtxt('data/pca_50.csv', delimiter=',')
#X = genfromtxt('train_data.csv', delimiter=',')

Y_0 = genfromtxt('train_labels.csv', delimiter=',')

print(X.shape)

Y = np.zeros((Y_0.shape[0], genres))


# Shuffle training data
numbers = np.arange(genres)

for i in range(Y_0.shape[0]):
	index = int(Y_0[i]) - 1
	numbers[index] += 1
	Y[i][index] = 1.0

ind = np.arange(X.shape[0])
np.random.shuffle(ind)
X = X[ind]
Y = Y[ind]
X.shape

# create model
model = Sequential()
model.add(Dense(30, input_dim=inputs, activation='relu', bias_initializer='ones'))
model.add(Dropout(0.7))
model.add(Dense(6, activation='relu', bias_initializer='ones'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu', bias_initializer='ones'))
model.add(Dense(genres, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split to train and verification data
X0 = X[:trainsize]
Y0 = Y[:trainsize]

X1 = X[trainsize:]
Y1 = Y[trainsize:]

# Fit model
model.fit(X0, Y0, epochs=2500, batch_size=3300)

# Evaluate the model
scores = model.evaluate(X1, Y1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
