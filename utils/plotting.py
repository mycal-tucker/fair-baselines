import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_encodings(list_of_points):
    pca = PCA(n_components=2)
    encs0, encs1 = list_of_points
    if encs0.shape[1] > 2:
        array0 = np.asarray(encs0)
        array1 = np.asarray(encs1)
        pca.fit(array0)
        encs0 = pca.transform(array0)
        encs1 = pca.transform(array1)
    fig, ax = plt.subplots()
    ax.scatter(encs0[:, 0], encs0[:, 1])
    ax.scatter(encs1[:, 0], encs1[:, 1])
    plt.show()
