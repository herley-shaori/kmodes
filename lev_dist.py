import numpy as np
from kmodes.kmodes import KModes

# random categorical data
# data = np.random.choice(20, (100, 10))

data = [[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[0,0,1,1,1],[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[0,0,1,1,1],[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[0,0,1,1,1],[1,1,1,1,1]]

km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

# Print the cluster centroids
print(km.cluster_centroids_)