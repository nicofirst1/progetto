from sklearn.decomposition import PCA
import numpy as np
from Preprocessing import *


def scoring(prediction):
    return np.mean(prediction == TEST_DATASET["sentiment"])


def pricipal_component(dataset):

    clean_x_train = sentences_polishing(list(dataset["review"]))

    x_train_vec = string2vecCV(clean_x_train, max_features=len(clean_x_train))[0]

    pca = PCA(n_components=min(TRAIN_DATASET.shape[0], TRAIN_DATASET.shape[1]))
    xtrain = pca.fit_transform(x_train_vec)

    print(xtrain.shape)
