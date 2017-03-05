import matplotlib.pyplot as plt
import time

from matplotlib.ticker import FormatStrFormatter
from sklearn.feature_selection import chi2
import operator


def plot_chi2(xtrain_vec, ytrain_vec, xtrain_str):

    print("calcolo del chi2...")
    start=time.time()
    chi2_res=chi2(xtrain_vec,ytrain_vec)[0]
    end=time.time()
    print("tempo impiegato: "+str(end-start)+" secodni")


def plot_vector(trans):

    # prendo i valori di tutte le parole
    idf=trans.idf_

    # # creo un dizionario
    # d=dict(zip(features, idf))
    # #scelgo solo le 100 parole con il punteggio migliore
    # sort=sorted(d.items(),key=operator.itemgetter(1),reverse=True)[:50]
    # # le ritrasformo in dizionario per essere plottate
    # d=dict(sort)

    # sorto i valori e creo una lista di tuple dove il primo elemnto è l'indice e il secondo il valore
    sort=sorted(idf)
    values=enumerate(sort)

    #trovo il minimo e massimo
    min_value=int(min(sort))
    max_value=int(max(sort))

    # creo un dizionario in cui le chiavi sono i valori compresi tra min e max, e i valori settati a zero per il momento
    values_dict=dict((elem,0) for elem in range(min_value,max_value+1))

    # conto quanti elemti nella lista si trovano nel range di valori
    for elem in sort:
        values_dict[int(elem)]+=1

    x,y=zip(*values)

    # setto la grandezza e il titolo
    plt.figure(figsize=(10,10))
    plt.title("All of 70708 words from TFIDF")

    # per ogni elemto nel dizioniario traccio una linea per le chiaivi e ci scrivo il valore
    for elem in values_dict.keys():
        plt.axhline(y=elem,c="red")
        plt.text(-10000,elem,str(values_dict[elem])+" words")


    plt.scatter(x,y)
   # plt.xticks(range(len(d)), d.keys(), rotation=90)
    plt.show()

