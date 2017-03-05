import os

import pandas as pd
import time
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation2, strip_non_alphanum, strip_tags
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

TEST_PATH = os.getcwd()+"/datasets/testDataLabeled.tsv"
TRAIN_PATH_LABLED = os.getcwd()+"/datasets/labeledTrainData.tsv"
TRAIN_PATH_UNLABLED=os.getcwd()+"/datasets/unlabeledTrainData.tsv"
# salvo i dati in formato dataframe
TEST_DATASET = pd.read_csv(TEST_PATH, header=0, sep="\t", quoting=3)
del TEST_DATASET["Unnamed: 0"]
TRAIN_DATASET_LABLED = pd.read_csv(TRAIN_PATH_LABLED, header=0, sep="\t", quoting=3)
TRAIN_DATASET_UNLABLED = pd.read_csv(TRAIN_PATH_UNLABLED, header=0, sep="\t", quoting=3)


DEBUG = False


def sentences_polishing(words_lst):



    # elimino i tag html
    words_lst = [strip_tags(x) for x in words_lst]

    # elimino le stopwords dalle frasi
    words_lst = [remove_stopwords(x) for x in words_lst]

    # elimino i punctuation dalle frasi
    words_lst = [strip_punctuation2(x) for x in words_lst]

    # elimino tutti i caratteri non alphanumerici
    words_lst = [strip_non_alphanum(x) for x in words_lst]

    # elimino le frasi vuote
    words_lst = [x for x in words_lst if x]

    # con questa pulizia del dataset sono passato da 156060 elementi a 102355

    return words_lst


def word_polishing_division(sentence_lst):

    # # creo una lista di liste che conterranno ogni parola, portanto le parole in minuscolo,
    # # adesso words_lst ha un'aspetto del tipo: [['il','cane','abbaia'],['il','gatto','miagola']],
    # # dove ongi parola di una frase è un elemento di una list ache va a comporre una lista di frasi
    words_lst = [x.lower().split() for x in sentence_lst]

    # elimino le parole con lunghezza minore di 3
    for elem in words_lst:
        for word in elem:
            if len(word) < 3:
                elem.remove(word)

    return words_lst


def string2vecCV(x_train_str, x_test_str=None, max_features=20000):
    """preso in input due Series (vettori) usa il principio della bag of word per creare un dizionario con le parole del
    train set e lo trasforma. In seguito trasforma anche il test set (se passato) e ritorna una tupla in cui il primo
    elemento è il train set vettorializzato e il secondo (sempre se presente) è il test set vettorializzato"""

    # pulisco l'x_test
    clean_x_train = sentences_polishing(list(x_train_str))

    # inizzializzo un CountVectorize che usa il prpincipio della bag of words per trasformare tutte le frasi del dataset
    #  in un datased multidimansionale dove ongi parola è rappresentata da un  valore numerico che indica le ripetizioni
    #  della stessa ll'interno della frase
    vect = CountVectorizer(analyzer="word", max_features=max_features)
    x_train_vec = vect.fit_transform(clean_x_train).toarray()

    if DEBUG:
        print(x_train_vec.shape)
        vocab = vect.get_feature_names()
        print(sorted(vocab, key=lambda elem: len(elem)))

    # se è presente il test set allora lo pulisco, lo trasformo e ritorno la tupla completa
    if x_test_str is not None:
        clean_x_test = sentences_polishing(list(x_test_str))
        x_test_vec = vect.transform(clean_x_test)
        return x_train_vec, x_test_vec
    else:  # altrimenti ritporno solo il test set nella tupla
        return x_train_vec, None


def string2vecTFIDF(x_train_str_labled,x_train_str_unlabled, x_test_str):

    # pulisco l'x_test uenndo labled e unlabled
    print("pulizia dell'xtrain")
    start = time.time()

    clean_x_trainL = sentences_polishing(list(x_train_str_labled))
    clean_x_trainU = sentences_polishing(list(x_train_str_unlabled))

    # inizzializzo un CountVectorize che usa il prpincipio della bag of words per trasformare tutte le frasi del dataset
    #  in un datased multidimansionale dove ongi parola è rappresentata da un  valore numerico che indica le ripetizioni
    #  della stessa ll'interno della frase
    vect = TfidfVectorizer(min_df=2,max_df=0.96,sublinear_tf = True,max_features = 200000)
    vect = vect.fit(clean_x_trainL+clean_x_trainU)
    x_train_vec=vect.transform(clean_x_trainL).toarray()
    print("xtrain trasformato")

    # se è presente il test set allora lo pulisco, lo trasformo e ritorno la tupla completa
    clean_x_test = sentences_polishing(list(x_test_str))
    x_test_vec = vect.transform(clean_x_test)

    x_train_str=vect.inverse_transform(x_train_vec)

    end = time.time()

    print("pulizia e trasformazione dell'xtest avvenuta\n tempo impiegato: "+str(end-start)+" secodni\n")
    return x_train_vec, x_test_vec, x_train_str

def dimensionality_reductionKB(xtrain,ytrain, xtest, percentage=0.9):

    print("inizio riduzione...")
    start = time.time()

    #scelgo quanti k mantenere
    k=int(len(xtrain)*percentage)
    kbest=SelectKBest(chi2,k=k)
    new_xtrain=kbest.fit_transform(xtrain,ytrain)
    print("xtrain ridotto!")
    new_xtest=kbest.transform(xtest)
    end = time.time()

    print("xtest ridotto!\n tempo impiegato: "+str(end-start)+" secondi\n")

    return new_xtrain,new_xtest