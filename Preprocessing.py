import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation2

TEST_PATH = "/Users/nicolo/PycharmProjects/progetto1/test.tsv"
TRAIN_PATH = "/Users/nicolo/PycharmProjects/progetto1/train.tsv"

# salvo i dati in formato dataframe
TEST_DATASET = pd.DataFrame.from_csv(TEST_PATH, sep='\t')
TRAIN_DATASET = pd.DataFrame.from_csv(TRAIN_PATH, sep='\t')

DEBUG = False


def dataset_inizialization():
    # creo una lista di frasi
    words_lst = list(TRAIN_DATASET["Phrase"])

    # elimino le stopwords dalle frasi
    words_lst = [remove_stopwords(x) for x in words_lst]

    # elimino i punctuation dalle frasi
    words_lst = [strip_punctuation2(x) for x in words_lst]

    # elimino le frasi vuote
    words_lst = [x for x in words_lst if x]

    # elimino le frasi doppione
    words_lst = list(set(words_lst))

    ## con questa pulizia del dataset sono passato da 156060 elementi a 102355

    # creo una lista di liste che conterranno ogni parola, portanto le parole in minuscolo,
    # adesso words_lst ha un'aspetto del tipo: [['il','cane','abbaia'],['il','gatto','miagola]],
    # dove ongi parola di una frase è un elemento di una list ache va a comporre una lista di frasi
    words_lst = [x.lower().split() for x in words_lst]

    # elimino le parole con lunghezza minore di 3
    for elem in words_lst:
        for word in elem:
            if (len(word) < 3): elem.remove(word)

    res_lst = list(TRAIN_DATASET["Sentiment"])

    return (words_lst, res_lst)


def prova1(words_lst, res_lst):
    # creo un dizionario di elementi chiave
    dict = corpora.Dictionary(words_lst)

    if (DEBUG):
        pr = sorted(list(dict.iteritems()), key=lambda tup: len(tup[1]))
        print(pr)
        print(dict)

    # trasformo il test set da stringa a vettore
    transformed_lst = []
    for elem in list(TRAIN_DATASET["Phrase"]):
        transformed_lst.append(dict.doc2bow(elem.lower().split()))

    if (DEBUG): print(transformed_lst)

    # inizzializzo il calssificatore
    forest = RandomForestClassifier(n_estimators=100)
    res_array = TRAIN_DATASET["Sentiment"]
    forest.fit(transformed_lst[0], res_array.iloc[0])


def prova2(words_lst, res_lst):
    # inizzializzo l'estimatore con la lista di parole, eliminando quelle che si ripetono eno di 11 volte

    model = Word2Vec(words_lst, min_count=10, size=200)
