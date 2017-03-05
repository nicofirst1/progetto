import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn.feature_selection import chi2
import operator
from math import isnan



def plot_chi2(xtrain_vec, ytrain_vec, trans):

    print("calcolo del chi2...")
    start=time.time()
    chi2_res=chi2(xtrain_vec,ytrain_vec)[0]
    end=time.time()
    print("tempo impiegato: "+str(end-start)+" secodni")

    #creo il dizionario e lo sorto
    features=trans.get_feature_names()
    d=dict(zip(features,chi2_res))

    #elimino i valori nan
    d ={k: d[k] for k in d if not isnan(d[k])}

    sort=sorted(d.items(),key=operator.itemgetter(1),reverse=False)[:20]
    keys=[i[0] for i in sort]
    values=[int(i[1]) for i in sort]

    # calcolo la media dei valori
    mean = 0
    for elem in values:
        mean += elem

    mean /= len(values)

    plt.figure()
    # setto il titolo
    plt.title("Worse 20 features for chi2")

    # plotto i dati
    plt.bar(range(len(sort)), values, align='edge',color="green")
    plt.xticks([(x+0.4) for x in range(len(sort))], keys, rotation=90, y=0.8, color="black", size="large")

    # plotto la media
    plt.axhline(y=mean, c="red")
    plt.text(0, mean, "mean: " + str(mean), color="red", size="large")

    plt.show()







def plot_top_n_words(trans, n, reverse=True):
    """Questa funzione prende in input 3 parametri, un trasformatore, un intero (si consiglia sotto i 50) e un boolean.
    La sua funzione è quella di plottare le n parole con punteggio piu altro (o piu basso dipende da reverse)"""

    # prendo le parole e i valori associati
    idf = trans.idf_
    features=trans.get_feature_names()


    # creo un dizionario
    d=dict(zip(features, idf))
    #scelgo solo le 100 parole con il punteggio migliore
    sort=sorted(d.items(),key=operator.itemgetter(1),reverse=reverse)[:n]
    # le ritrasformo in dizionario per essere plottate
    d=dict(sort)

    #calcolo la media dei valori
    mean=0
    for elem in d.values():
        mean+=elem

    mean/=len(d)



    plt.figure()
    #scelgo il titolo piu opportuno
    if(reverse): plt.title("Top "+str(n)+" features")
    else:  plt.title("Worst "+str(n)+" features")

    # plotto i dati
    plt.bar(range(len(d)), d.values(), align='center')
    plt.xticks(range(len(d)), d.keys(), rotation=90,y=0.5, color="white")

    #plotto la media
    plt.axhline(y=mean, c="red")
    plt.text(0,mean,"mean: "+str(mean),color="red",size="large")

    plt.show()


def plot_vector(trans):
    """Questa funzione prende in input un trasformatore( TFIDF o countVectorize), ne estrae i valori e li plotta
    dividento le ordinate in n parti, con n preso tra minimo e massimo tra i valori presenti. Inoltre stampe le parole
     presenti in ogni range di n valori"""

    # prendo i valori di tutte le parole
    idf=trans.idf_


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

