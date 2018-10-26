from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from init import *

train_data = data

train_data = StandardScaler().fit_transform(train_data)

pca = PCA(n_components=30)

# Shape (4363, 2)
principalComponents = pca.fit_transform(train_data)
