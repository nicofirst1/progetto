import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation2, strip_non_alphanum, strip_tags
from sklearn.feature_extraction.text import CountVectorizer

TEST_PATH = "/Users/nicolo/PycharmProjects/progetto1/testData.tsv"
TRAIN_PATH = "/Users/nicolo/PycharmProjects/progetto1/labeledTrainData.tsv"

# salvo i dati in formato dataframe
TEST_DATASET = pd.DataFrame.from_csv(TEST_PATH, sep='\t')
TRAIN_DATASET = pd.DataFrame.from_csv(TRAIN_PATH, sep='\t')

DEBUG = False


def word_polishing(words_lst):

    # elimino i tag html
    words_lst=[strip_tags(x) for x in words_lst]

    # elimino le stopwords dalle frasi
    words_lst = [remove_stopwords(x) for x in words_lst]

    # elimino i punctuation dalle frasi
    words_lst = [strip_punctuation2(x) for x in words_lst]

    #elimino tutti i caratteri non alphanumerici
    words_lst=[strip_non_alphanum(x) for x in words_lst]

    # elimino le frasi vuote
    words_lst = [x for x in words_lst if x]

    # elimino le frasi doppione
    #words_lst = list(set(words_lst))

    ## con questa pulizia del dataset sono passato da 156060 elementi a 102355

    # # creo una lista di liste che conterranno ogni parola, portanto le parole in minuscolo,
    # # adesso words_lst ha un'aspetto del tipo: [['il','cane','abbaia'],['il','gatto','miagola']],
    # # dove ongi parola di una frase è un elemento di una list ache va a comporre una lista di frasi
    # words_lst = [x.lower().split() for x in words_lst]
    #
    # # elimino le parole con lunghezza minore di 3
    # for elem in words_lst:
    #     for word in elem:
    #         if (len(word) < 3): elem.remove(word)
    #
    # res_lst = list(TRAIN_DATASET["Sentiment"])

    return words_lst


def prova1(words_lst, res_lst):
    # creo un dizionario di elementi chiave
    dict = corpora.Dictionary(words_lst)

    if (DEBUG):
        pr = sorted(list(dict.iteritems()), key=lambda tup: len(tup[1]))
        print(pr)
        print(dict)

    # trasformo il train set da stringa a lista di liste
    transformed_lst = [dict.doc2bow(x.lower().split()) for x in list(TRAIN_DATASET["Phrase"])]

    # dato che l'estimatore random forest tree accetta come X array o matrici sparse trasformo
    #  la lista di liste in una matrice sparsa
    sparse=csr_matrix(transformed_lst)


    # inizzializzo il calssificatore
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(sparse, res_lst)

def prova2(words_lst, res_lst):
    print("start prova2")


    #inizzializzo un CountVectorize che usa il prpincipio della bag of words per trasformare tutte le frasi del dataset
    #  in un datased multidimansionale dove ongi parola è rappresentata da un  valore numerico che indica le ripetizioni
    #  della stessa ll'interno della frase
    vect=CountVectorizer(analyzer = "word")

    X_train=vect.fit_transform(words_lst).toarray()
    print("end vect")


    if DEBUG:
        print(X_train.shape)
        vocab = vect.get_feature_names()
        print(sorted(vocab, key=lambda elem: len(elem)))

    #inizzializzo l'estimatore e inizio il fittaggio
    forest = RandomForestClassifier(n_estimators=100,n_jobs=-1)
    forest=forest.fit(X_train,res_lst)
    print("end fit")


    # uso lo stesso principio di prima per trasformare (solo trasformare) il train dataset
    #  in un vettore multidimensionale con lo STESSO CountVectorizer, prima devo pulirlo

    clean_test=word_polishing(list(TEST_DATASET["review"]))
    clean_vect_test=vect.transform(clean_test).toarray()
    print("end vect2")


    #adesso posso provare a fare la predizione
    pred=forest.predict(clean_vect_test)
    print("end pred")

    #salvo i risultati in un dataframe e li trasformo in csv
    to_save=pd.DataFrame(data={"id":TEST_DATASET["id"],"sentiment":pred})
    to_save.to_csv(path_or_buf="/Users/nicolo/PycharmProjects/progetto1/pred.csv")






