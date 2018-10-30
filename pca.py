from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from init import *

train_data = data
components = 50

train_data = StandardScaler().fit_transform(train_data)

pca = PCA(n_components=components)

# Shape (4363, # of components)
principalComponents = pca.fit_transform(train_data)

np.savetxt('data/pca_' + str(components) + '.csv',principalComponents, delimiter=',')
