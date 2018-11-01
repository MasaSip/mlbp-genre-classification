# Based on http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

from sklearn.preprocessing import StandardScaler

from init import *


X = StandardScaler().fit_transform(data)
y = labels
# reduce shape of labels from (n,1) to (n,)
y.shape = (y.shape[0],)
n_samples, n_features = X.shape
n_neighbors = 30


# ---------------------------------------------------------------------
# Find corresponding images for visualization
digits = datasets.load_digits(n_class=10)
image_loc = np.zeros(11)
for i in range(10):
    image_loc[i] = np.where(digits.target==i)[0][0]
image_loc[10] = image_loc[0]
image_loc = image_loc.astype(int)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    plt.text(X[1, 0], X[1, 1], str(y[1]),
                 color=plt.cm.Set1(y[1] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[image_loc[y[i]]], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
print(X_projected)
plot_embedding(X_projected, "Random Projection of the digits")



plt.show()
