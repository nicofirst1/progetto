import pandas as pd
import pickle
import time
from gensim.scripts import word2vec_standalone
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation2, strip_non_alphanum, strip_tags
from sklearn.feature_extraction.text import CountVectorizer

TEST_PATH = "/Users/Stefan/PycharmProjects/progetto/testDataLabeled.tsv"
TRAIN_PATH = "/Users/Stefan/PycharmProjects/progetto/labeledTrainData.tsv"

# salvo i dati in formato dataframe
TEST_DATASET = pd.read_csv(TEST_PATH, header=0, sep="\t", quoting=3)
del TEST_DATASET["Unnamed: 0"]
TRAIN_DATASET = pd.read_csv(TRAIN_PATH, header=0, sep="\t", quoting=3)

DEBUG = False


def sentences_polishing(words_lst):

    # elimino i tag html
    words_lst=[strip_tags(x) for x in words_lst]

    # elimino le stopwords dalle frasi
    words_lst = [remove_stopwords(x) for x in words_lst]

    # elimino i punctuation dalle frasi
    words_lst = [strip_punctuation2(x) for x in words_lst]

    # elimino tutti i caratteri non alphanumerici
    words_lst=[strip_non_alphanum(x) for x in words_lst]

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


def string2vecCV(x_train_str ,x_test_str=None, max_features=20000):
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


def word2vec(x_train_str,x_test_str=None):
    # prima di tutto eseguo la pulizia delle frasi e poi delle parole, dato che l'estimatore Word2Vec riceve in input
    # una lista di liste di prole che rappresentano una frase
    clean_x_train_words = word_polishing_division(sentences_polishing(x_train_str))
    clean_x_test_words = word_polishing_division(sentences_polishing(x_test_str))

    model = Word2Vec(clean_x_train_words, min_count=10, workers=-1)
def cluster_dict_syn(word_for_cluster=100):
    """questa funzioen prende in input un parametro facoltativo e fa le seguenti operazioni:
    carica il modello word2vec di google (opportunamente ottimizzato) su cui è stato effettuato un train con 3 milioni di parole;
    divide le parole in cluster per similitudine semantica, tramite un analisi a k-means (il numero di parole per cluster
     è dettato dal parametro facoltativo);
     ritorna un dizionario in cui ogni parola è associata ad un indice di cluster
    """

    # se il dizionario è gia stato creato e salvato allora ritorno quello, altrimenti procedo con l'esecuzione
    pickle_name="kmeans"+str(word_for_cluster)+".pickle"

    if(pickle_name in os.listdir("/k-means/")):
        with open("/k-means/"+pickle_name) as file:
            dictionary=pickle.load(file)
            return dictionary


    print("Inizio caricamento modello word2vec...")
    #carico il modello pre-trained di google (il modello Faster l'ho creato apposta per la velocità usando la funzione
    # init_sims(replace=True)che sfrutta la memria RAM)
    model = Word2Vec.load('Faster', mmap='r')
    model.syn0norm = model.syn0

    #per collegare questo modello all'esercizio mi devo suddividere le parole per similarità semantica, faccio questo
    # tramite un semplice clustering di parole. Per trovare i vari centroidi dei cluster semantici uso il k-mean fornito
    # da sklearn

    word_vect=model.syn0

    #devo decidere le dimensioni del km, ovvero quante parole sono presenti in un cluster, per ora mi tengo sulle 100 parole

    cluster_word_num=int(word_vect.shape[0]/word_for_cluster)

    #prendo il tempo impiegto dal km (solo per ragioni di speedup)
    start = time.time()
    print("Inizio fittaggio del k-mean...")

    #creo il km con il massimo della parallelizzazione e procedo nel fittaggio con predizione
    km=KMeans(n_clusters=cluster_word_num, n_jobs=-1, verbose=1)
    # cluster_index sara un vettore di indici relativi ai cluster
    cluster_index=km.fit_predict(word_vect)

    end=time.time()
    print("Il km ha impiegato "+(end-start)+ " secondi")

    # creo un dizionario in cui ogni parola è associata ad un cluster
    dictionary=dict(zip(model.index2word,cluster_index))
    print("Fine creazione dizionario")

    # per motivi di convenienza salvo il dizionario usando pickle
    with open("/k-means"+pickle_name,'wb') as file:
        pickle.dump(dictionary,file,protocol=pickle.HIGHEST_PROTOCOL)

    return dictionary









