import os
import pandas as pd
import time
from nltk.stem import  SnowballStemmer,WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation2, strip_non_alphanum, strip_tags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from os import system

#apertura e aslvataggio dei vari datasets
TEST_PATH = os.getcwd()+"/datasets/testDataLabeled.tsv"
TRAIN_PATH_LABLED = os.getcwd()+"/datasets/labeledTrainData.tsv"
TRAIN_PATH_UNLABLED=os.getcwd()+"/datasets/unlabeledTrainData.tsv"
# salvo i dati in formato dataframe
TEST_DATASET = pd.read_csv(TEST_PATH, header=0, sep="\t", quoting=3)
del TEST_DATASET["Unnamed: 0"]
TRAIN_DATASET_LABLED = pd.read_csv(TRAIN_PATH_LABLED, header=0, sep="\t", quoting=3)
TRAIN_DATASET_UNLABLED = pd.read_csv(TRAIN_PATH_UNLABLED, header=0, sep="\t", quoting=3)



def sentences_polishing(words_lst, what):


    lst_len_start=sum(len(s) for s in words_lst)
    print("Pulizia della lista con "+str(lst_len_start)+" chars, per "+what)

    # elimino i tag html
    words_lst = [strip_tags(x) for x in words_lst]

    # elimino le stopwords dalle frasi
    words_lst = [remove_stopwords(x) for x in words_lst]

    # elimino i punctuation dalle frasi
    words_lst = [strip_punctuation2(x) for x in words_lst]

    # elimino tutti i caratteri non alphanumerici
    words_lst = [strip_non_alphanum(x) for x in words_lst]

    #creo una lista di recensioni vuote
    new_review_lst=[]

    #per ogni recensione applico una funzione e aggiungo il risultato alla lista
    for review in words_lst:
        new_review=stemming_lemmatization(review)
        new_review_lst.append(new_review)


    # elimino le frasi vuote
    words_lst = [x for x in words_lst if x]


    lst_len_end=sum(len(s) for s in words_lst)
    cleaned=lst_len_start-lst_len_end
    print("Puliti "+str(cleaned)+" ("+str(int(cleaned/lst_len_start*100))+"%) chars, per "+what+"\n")

    return words_lst

def stemming_lemmatization(review):

    #Inizzializzo lo stemmer e il lemmatizer
    stemmer=SnowballStemmer("english")
    lemmatizer=WordNetLemmatizer()

    # separo le parole per spazio
    words_list=review.split()

    #elimino le parole  con meno di tre char
    words_list=[x for x in words_list if len(x)>2]

    #applico lo stemmer e il lemmatizer
    words_list=[stemmer.stem(x) for x in words_list]
    words_list=[lemmatizer.lemmatize(x) for x in words_list]


    return " ".join(words_list)



def string2vecTFIDF(x_train_str_labled,x_train_str_unlabled, x_test_str):

    # pulisco l'x_test uenndo labled e unlabled
    print("pulizia del train dataset\n")
    start = time.time()

    clean_x_trainL = sentences_polishing(list(x_train_str_labled),"XtrainLabled")
    clean_x_trainU = sentences_polishing(list(x_train_str_unlabled),"XTrainUnlabled")

    # inizzializzo un CountVectorize che usa il prpincipio della bag of words per trasformare tutte le frasi del dataset
    #  in un datased multidimansionale dove ongi parola è rappresentata da un  valore numerico che indica le ripetizioni
    #  della stessa ll'interno della frase
    vect = TfidfVectorizer(min_df=5,max_df=0.80,sublinear_tf = True,max_features = 85000,strip_accents="ascii",
                           ngram_range=(1,3))
    """max_features=200000->89184
    max_features=250000->89108
    max_features=100000->89348
    max_features=50000->8916
    max_features=75000->8928
    max_features=85000, max_df=0.8->89352
    max_df=0.9->89352"""


    vect = vect.fit(clean_x_trainL+clean_x_trainU)
    x_train_vec=vect.transform(clean_x_trainL).toarray()
    print("train dataset trasformato, creato dizionario con "+str(len(vect.vocabulary_))+" parole")

    # se è presente il test set allora lo pulisco, lo trasformo e ritorno la tupla completa
    clean_x_test = sentences_polishing(list(x_test_str),"XTest")
    x_test_vec = vect.transform(clean_x_test)

    end = time.time()
    tot=end-start

    print("pulizia e trasformazione del test dataset avvenuta\ntempo impiegato: "+str(int(tot/60))+"'"+str(int(tot%60))+"''\n")
    system('say "Pulizia del dataset avvenuta"')


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
    tot=end-start

    print("xtest ridotto!\ntempo impiegato: "+str(int(tot/60))+"'"+str(int(tot%60))+"''\n")
    system('say "Test del ki quadro completato"')


    return new_xtrain,new_xtest