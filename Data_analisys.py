import numpy as np
from scipy.stats import stats

from Preprocessing import *
import matplotlib.pyplot as plt




def scoring(prediction):
    return np.mean(prediction == TEST_DATASET["sentiment"])


def plot_chi2(xtrain_vec, ytrain_vec, xtrain_str):

    print("calcolodel chi2...")
    start=time.time()
    chi2_res=chi2(xtrain_vec,ytrain_vec)[0]
    end=time.time()
    print("tempo impiegato: "+str(end-start)+" secodni")

    plt.figure(figsize=(6, 6))
    wscores = zip(xtrain_str, chi2_res)
    wchi2 = sorted(wscores, key=lambda x: x[1])
    topchi2 = wchi2[-25:]
    x = range(len(topchi2[1]))
    labels = topchi2[0]
    plt.barh(x, topchi2[1], align='center', alpha=.2, color='g')
    plt.plot(topchi2[1], x, '-o', markersize=2, alpha=.8, color='g')
    plt.yticks(x, labels)
    plt.xlabel('$\chi^2$')
    plt.show()




