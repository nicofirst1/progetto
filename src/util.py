from os import system

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from src.Preprocessing import TEST_DATASET, string2vecTFIDF, dimensionality_reductionKB


# Todo: cross validation per tutning dei parametri


def cross_validation_score(model, x):
    scores = cross_val_score(model, x, y=TEST_DATASET["sentiment"])
    print(scores)

def save_to_csv(pred):

    # salvo il risultato di una predizione in formato csv per essere spedito a kaggle
    dict={"id":TEST_DATASET["id"],"sentiment":pred}
    output=pd.DataFrame(data=dict)
    output.to_csv("res.csv", index=False, quoting=3)

def scoring(prediction, true,what):

    #prendo la media dell'errore tra valori predetti e reali
    pred = str(np.mean(prediction == true) * 100)
    #stampo il risultato
    print("il risultato della predizione per "+what+" è: " + pred)
    string = "Risultato ottenuto pari a " + pred + " per cento"
    system('say ' + string)


def polish_tfidf_kbest(train_set_labled, train_set_unlabled, test_set):
    # faccio le divisioni
    xtrainL = train_set_labled["review"]
    xtrainU = train_set_unlabled["review"]
    xtest = test_set["review"]
    ytrain = train_set_labled["sentiment"]

    print("inizio trasformazione da stringa a vettore...")

    # trasformo da stringhe a vettori
    xtrain_vec, xtest_vec, vect = string2vecTFIDF(xtrainL, xtrainU, xtest)

    feature_names = vect.get_feature_names()

    print("inizio test del chi2...")

    # eseguo una ricerca delle labels migliori
    reduced_xtrain_vec, reduced_xtest_vec = dimensionality_reductionKB(xtrain_vec, ytrain, xtest_vec,feature_names)
    return reduced_xtrain_vec, reduced_xtest_vec, ytrain, feature_names


def svm_grid(xtrain, ytrain):
    svm = RandomForestClassifier(n_jobs=-1, verbose=1, criterion="entropy")

    param = {"n_estimators": [50, 100, 150, 200, 250, 300]}

    grid = GridSearchCV(svm, param, n_jobs=-1, verbose=1, error_score=-1)
    grid.fit(xtrain, ytrain)

    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)

def save_wrong_answer(pred,names,xtest):

    # n è il numero di feature che volgio stampare per ogni predizione sbagliata
    n=3


    # salvo i veri valori della predizione, e creo una lista vuota
    true=TEST_DATASET["sentiment"]
    wrong=[]

    idx=0
    index=[]

    # per ogni coppia di valori (predizione,vero valore)
    for i,j in zip(pred,true):
        # se i due dati non coincidono
        if (i!=j):
            # aggiungo l'i-esima review nella lista wrong, collegata alla predizione effettuata dal modello
            wrong.append((idx,TEST_DATASET["review"].iget(idx),i))
            # e savlo anche l'inidce in cui si trova la predizione sbalgiata
            index.append(idx)
        idx+=1


    # creo un file in cui salvo i dati per analizzarli successivamente, questo file conterrà tutte le recensione
    #  erroneamente classificate
    with open("/Users/nicolo/PycharmProjects/progetto/wrong.txt","w") as file:
        file.write('\n\n'.join('%d) %s \t Incorrect= %s' % x for x in wrong))

    print("wrong file creato")

    # prendo tutte le recensioni classificate erronemanete dall'xtsts
    samples=[]
    for elem in index:
        samples.append(xtest[elem])

    # per ogni recensione prendo le features corrispondenti
    correct=[]
    for line in samples:
        # collego le features alle parole
        line=[y for y in zip(line,names)]
        # prendo le top n parole
        line=sorted(line,key= lambda tup: tup[0])[-n:]
        # le aggiungo alla lista
        correct.append(line)


    anlayzer=[]
    idx=0
    # analyzer avra la seguente forma (indice, (valore della features, parola associata))
    for elem in index:
        anlayzer.append((elem,correct[idx]))
        idx+=1

        # creo un file in cui salvo i dati per analizzarli successivamente
    with open("/Users/nicolo/PycharmProjects/progetto/wrong_data.txt", "w") as file:
        file.write('\n\n'.join('%d) %s' % x for x in anlayzer))

    print("wrong_data file creato")



