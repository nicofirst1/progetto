import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from gensim import corpora

TEST_PATH="/Users/nicolo/PycharmProjects/progetto1/test.tsv"
TRAIN_PATh="/Users/nicolo/PycharmProjects/progetto1/train.tsv"

TEST_DATASET=False
TRAIN_DATASET=False

DEBUG=False


def dataset_inizialization(test_path, train_path):

    global TEST_DATASET, TRAIN_DATASET

    #salvo i dati in formato dataframe
    TEST_DATASET=pd.DataFrame.from_csv(test_path, sep='\t')
    TRAIN_DATASET=pd.DataFrame.from_csv(train_path, sep='\t')


    #elimino le frasi con solo parole non rilevanti
    stop_words='for a of the and to in it not may'.split()
    words_lst=[x for x in list(TRAIN_DATASET["Phrase"]) if x not in stop_words]

    # elimino i caratteri non alfabetici
    words_lst=[x for x in words_lst if x.isalpha()]

    # creo una lista di liste portanto le parole in minuscolo
    words_lst=[x.lower().split() for x in words_lst]

    # elimino le parole con lunghezza minore di 3
    for elem in words_lst:
        for word in elem:
            if(len(word)<3): elem.remove(word)

    res_lst=list(TEST_DATASET["Sentiment"])

    return (words_lst, res_lst)


def prova1(words_lst, res_lst):

    # creo un dizionario di elementi chiave
    dict=corpora.Dictionary(words_lst)

    if(DEBUG):
        pr=sorted(list(dict.iteritems()),key= lambda tup: len(tup[1]))
        print(pr)
        print(dict)

    #trasformo il test set da stringa a vettore
    transformed_lst=[]
    for elem in list(TRAIN_DATASET["Phrase"]):
        transformed_lst.append(dict.doc2bow(elem.lower().split()))

    if(DEBUG): print(transformed_lst)

    # inizzializzo il calssificatore
    forest=RandomForestClassifier(n_estimators=100)
    res_array=TRAIN_DATASET["Sentiment"]
    forest.fit(transformed_lst[0],res_array.iloc[0])



