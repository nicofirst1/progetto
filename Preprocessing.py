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
TEST_DATASET = pd.read_csv(TEST_PATH, header=0, sep="\t", quoting=3)
TRAIN_DATASET = pd.read_csv(TRAIN_PATH, header=0, sep="\t", quoting=3)

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

def string2vecCV(x_train_str ,x_test_str=None):
    """preso in input due Series (vettori) usa il principio della bag of word per creare un dizionario con le parole del
    train set e lo trasforma. In seguito trasforma anche il test set (se passato) e ritorna una tupla in cui il primo
    elemento è il train set vettorializzato e il secondo (sempre se presente) è il test set vettorializzato"""


    # pulisco l'x_test
    clean_x_train=word_polishing(list(x_train_str))

    #inizzializzo un CountVectorize che usa il prpincipio della bag of words per trasformare tutte le frasi del dataset
    #  in un datased multidimansionale dove ongi parola è rappresentata da un  valore numerico che indica le ripetizioni
    #  della stessa ll'interno della frase
    vect=CountVectorizer(analyzer = "word", max_features = 10000)
    x_train_vec=vect.fit_transform(clean_x_train).toarray()


    if DEBUG:
        print(x_train_vec.shape)
        vocab = vect.get_feature_names()
        print(sorted(vocab, key=lambda elem: len(elem)))

    # se è presente il test set allora lo pulisco, lo trasformo e ritorno la tupla completa
    if x_test_str:
        clean_x_test=word_polishing(list(x_test_str))
        x_test_vec=vect.transform(clean_x_test)
        return x_train_vec, x_test_vec
    else: #altrimenti ritporno solo il test set nella tupla
        return x_train_vec, None









