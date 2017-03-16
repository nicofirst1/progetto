import os
import pandas as pd
import time
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation2, strip_non_alphanum, strip_tags
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apertura e aslvataggio dei vari datasets
TEST_PATH = os.getcwd()+"/datasets/testDataLabeled.tsv"
TRAIN_PATH_LABLED = os.getcwd()+"/datasets/labeledTrainData.tsv"
TRAIN_PATH_UNLABLED=os.getcwd()+"/datasets/unlabeledTrainData.tsv"
# salvo i dati in formato dataframe
TEST_DATASET = pd.read_csv(TEST_PATH, header=0, sep="\t", quoting=3)
del TEST_DATASET["Unnamed: 0"]
TRAIN_DATASET_LABLED = pd.read_csv(TRAIN_PATH_LABLED, header=0, sep="\t", quoting=3)
TRAIN_DATASET_UNLABLED = pd.read_csv(TRAIN_PATH_UNLABLED, header=0, sep="\t", quoting=3)




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


def string2vecTFIDF(x_train_str_labled,x_train_str_unlabled, x_test_str):

    # pulisco l'x_test uenndo labled e unlabled
    print("pulizia dell'xtrain")
    start = time.time()

    clean_x_trainL = sentences_polishing(list(x_train_str_labled))
    clean_x_trainU = sentences_polishing(list(x_train_str_unlabled))

    # inizzializzo un CountVectorize che usa il prpincipio della bag of words per trasformare tutte le frasi del dataset
    #  in un datased multidimansionale dove ongi parola è rappresentata da un  valore numerico che indica le ripetizioni
    #  della stessa ll'interno della frase
    vect = TfidfVectorizer(min_df=2,max_df=0.96,sublinear_tf = True,max_features = 200000,strip_accents="ascii")
    vect = vect.fit(clean_x_trainL+clean_x_trainU)
    x_train_vec=vect.transform(clean_x_trainL).toarray()
    print("xtrain trasformato")

    # se è presente il test set allora lo pulisco, lo trasformo e ritorno la tupla completa
    clean_x_test = sentences_polishing(list(x_test_str))
    x_test_vec = vect.transform(clean_x_test)

    end = time.time()

    print("pulizia e trasformazione dell'xtest avvenuta\ntempo impiegato: "+str(end-start)+" secondi\n")
    return x_train_vec, x_test_vec, vect

def dimensionality_reductionKB(xtrain,ytrain, xtest, percentage=0.85):

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