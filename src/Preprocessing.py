import os
import pandas as pd
import time
import numpy as np
from nltk.stem import SnowballStemmer
from gensim.parsing.preprocessing import  strip_punctuation2, strip_non_alphanum, strip_tags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from os import system
from nltk.corpus import stopwords
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

# Todo: stemming, stopwords


# apertura e salvataggio dei vari datasets

TEST_PATH = os.getcwd() + "/datasets/testDataLabeled.tsv"
TRAIN_PATH_LABLED = os.getcwd() + "/datasets/labeledTrainData.tsv"
TRAIN_PATH_UNLABLED = os.getcwd() + "/datasets/unlabeledTrainData.tsv"
# salvo i dati in formato dataframe
TEST_DATASET = pd.read_csv(TEST_PATH, header=0, sep="\t", quoting=3)
del TEST_DATASET["Unnamed: 0"]
TRAIN_DATASET_LABLED = pd.read_csv(TRAIN_PATH_LABLED, header=0, sep="\t", quoting=3)
TRAIN_DATASET_UNLABLED = pd.read_csv(TRAIN_PATH_UNLABLED, header=0, sep="\t", quoting=3)



def spot_differences(old, new, what, words=None):

    i=0
    j=0
    ris=""
    print("differenze per: "+what)
    for old_row, new_row in zip(old,new):
        for old_cell, new_cell in zip(old_row,new_row):
            if(old_cell!= new_cell) and (words):
                ris+="differenza per cella ["+str(i)+", "+str(j)+", "+words[j]+" ]\t"+str(old_cell)+" -> "+str(new_cell)+"\n"
            elif(old_cell!= new_cell):
                ris+="differenza per cella ["+str(i)+", "+str(j)+" ]\t"+str(old_cell)+" -> "+str(new_cell)+"\n"

            j+=1

        i+=1
    with open("difference_"+what+".txt","w") as file:
        file.write(ris)



def sentences_polishing(words_lst, what, deep_polishing=True, essential=False):

    # caclolo e printo il numero di char presenti in tutta la lista
    lst_len_start = sum(len(s) for s in words_lst)
    print("Pulizia della lista con " + str(lst_len_start) + " chars, per " + what)

    # elimino i tag html
    words_lst = [strip_tags(x) for x in words_lst]

    if not essential:

        # elimino i punctuation dalle frasi
        words_lst = [strip_punctuation2(x) for x in words_lst]

        # elimino tutti i caratteri non alphanumerici
        words_lst = [strip_non_alphanum(x) for x in words_lst]

    if deep_polishing:
        # inizzializzo un pool per il multiprocessing
        pool = Pool(processes=10)

        # per ogni recensione applico una funzione e aggiungo il risultato alla lista
        words_lst = pool.map(stemming_stopWords, words_lst)
        pool.close()
        pool.join()

    # elimino le frasi vuote
    words_lst = [x for x in words_lst if x]

    # ricalcolo il numero di char della lista e faccio la differenza con quello iniziale
    lst_len_end = sum(len(s) for s in words_lst)
    cleaned = lst_len_start - lst_len_end
    print("Puliti " + str(cleaned) + " (" + str(int(cleaned / lst_len_start * 100)) + "%) chars, per " + what + "\n")

    return words_lst


def stemming_stopWords(review):
    # Inizzializzo lo stemmer
    stemmer = SnowballStemmer("english")

    # separo le parole per spazio
    words_list = review.split()

    # elimino le parole  con meno di tre char
    words_list = [x for x in words_list if len(x) > 2]

    # rimuovo le stopwords
    words_list = [word for word in words_list if word not in stopwords.words('english')]

    # applico lo stemmer e il lemmatizer
    words_list = [stemmer.stem(x) for x in words_list]

    # rimuovo i numeri
    words_list = [word for word in words_list if word.isalpha()]

    return " ".join(words_list)


def string2vecTFIDF(x_train_str_labled, x_train_str_unlabled, x_test_str):
    print("pulizia del train dataset\n")
    start = time.time()

    # faccio la pulizia di ogni dataset
    clean_x_trainL = sentences_polishing(list(x_train_str_labled), "XtrainLabled")
    clean_x_trainU = sentences_polishing(list(x_train_str_unlabled), "XTrainUnlabled")
    clean_x_test = sentences_polishing(list(x_test_str), "XTest")

    # inizzializzo un CountVectorize che usa il prpincipio della bag of words per trasformare tutte le frasi del dataset
    #  in un datased multidimansionale dove ongi parola è rappresentata da un  valore numerico che indica le ripetizioni
    #  della stessa ll'interno della frase
    vect = TfidfVectorizer(min_df=5, max_df=0.80, sublinear_tf=True, max_features=85000,
                           strip_accents='unicode', token_pattern=r'\w{1,}',
                           ngram_range=(1, 2))

    # fitto il tfidf con il train labled e unlabled
    vect = vect.fit(clean_x_trainL + clean_x_trainU)
    # trasformo il tain labled
    x_train_vec = vect.transform(clean_x_trainL).toarray()
    print("train dataset trasformato, creato dizionario con " + str(len(vect.vocabulary_)) + " parole")

    # trasformo anche il testset e il validation set
    x_test_vec = vect.transform(clean_x_test).toarray()

    # prendo il tempo e stampo il risultato
    end = time.time()
    tot = end - start

    print("pulizia e trasformazione del test dataset avvenuta\ntempo impiegato: " + str(int(tot / 60)) + "' " + str(
        int(tot % 60)) + "''\n")
    system('say "Pulizia del dataset avvenuta"')

    return x_train_vec, x_test_vec, vect


def dimensionality_reductionKB(xtrain, ytrain, xtest,names, percentage=0.85):
    print("inizio riduzione...")
    start = time.time()

    # scelgo quanti k mantenere
    k = int(len(xtrain) * percentage)
    kbest = SelectKBest(chi2, k=k)
    new_xtrain = kbest.fit_transform(xtrain, ytrain)
    print("xtrain ridotto!")
    new_xtest = kbest.transform(xtest)
    end = time.time()
    tot = end - start

    spot_differences(xtrain,new_xtrain,"xtrain per chi2",words=names)

    print("xtest ridotto!\ntempo impiegato: " + str(int(tot / 60)) + "' " + str(int(tot % 60)) + "''\n")
    system('say "Test del ki quadro completato"')

    return new_xtrain, new_xtest
