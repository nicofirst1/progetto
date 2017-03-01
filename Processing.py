import pandas as pd
import numpy as np
from gensim.parsing import strip_punctuation2
from gensim.parsing import strip_tags
from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import TfidfTransformer
from Data_analisys import scoring
from sklearn.feature_extraction.text import CountVectorizer

from Preprocessing import  TEST_DATASET, string2vecCV, TRAIN_DATASET


def semantic_cluster_assignment(cluster_dictionary):

    #prima di tutto prendo le recensioni e le pulisco dai tah html e dalla punteggiatura
    review_lst=list(TRAIN_DATASET["review"])
    review_lst=[strip_tags(x) for x in  review_lst]
    review_lst=[strip_punctuation2(x) for x in  review_lst]

    print("Creazione dataset vuoto")
    # adesso creo un dataframe vuoto con review_lst.lenght righe
    index=range(0,len(review_lst)+1)
    new_dataset=pd.DataFrame(index=index)

    print("Inizio iterazione sul dataset")
    for sentence in review_lst:
        new_dataset.append(auxiliary(sentence,cluster_dictionary))


def auxiliary(sentence,dictionary):

    # splitto la frase separando le parole
    word_lst=sentence.tolower().split()

    #prealloco un array di zeri per ottimizzare il procedimento, questo array avra dimensione pari al numero dei cluster
    #  e sarà una riga del mio dataset composta dai vari labels
    array=np.zeros(dictionary)





def forest_classifier(train_dataset, test_dataset):

    # prima di tutto salvo le y_train
    y_train = train_dataset["sentiment"]

    # poi mando il train datasaet nel preprocessing usando il CountVectorizer
    x_train, x_test = string2vecCV(train_dataset["review"], x_test_str=test_dataset["review"])

    # inizzializzo l'estimatore e inizio il fittaggio
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train)
    x_train_tf = tf_transformer.transform(x_train)

    forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=1, criterion="entropy")
    forest = forest.fit(x_train, y_train)

    # adesso posso provare a fare la predizione
    pred_forest = forest.predict(x_test)

    # #salvo i risultati in un dataframe e li trasformo in csv
    # to_save=pd.DataFrame(data={"id":TEST_DATASET["id"],"sentiment":pred})
    # to_save.to_csv("pred.csv", index=False, quoting=3 )

    forest_score = scoring(pred_forest)
    print(forest_score)


def naive_bayes(train_dataset, test_dataset):

    y_train = train_dataset["sentiment"]
    x_train, x_test = string2vecCV(train_dataset["review"], x_test_str=test_dataset["review"])

    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train)
    x_train_tf = tf_transformer.transform(x_train)

    bernoulli = BernoulliNB()
    bernoulli = bernoulli.fit(x_train, y_train)
    pred_bernoulli = bernoulli.predict(x_test.toarray())
    print("BernoulliNB senza TfidTransformer: ")
    bst = scoring(pred_bernoulli)
    print(bst)

    multinomial = MultinomialNB()
    multinomial = multinomial.fit(x_train, y_train)
    pred_multinomial = multinomial.predict(x_test.toarray())
    print("MultinomialNB senza TfidTransformer: ")
    mst = scoring(pred_multinomial)
    print(mst)

    bernoulli = bernoulli.fit(x_train_tf, y_train)
    pred_bernoulli_tf = bernoulli.predict(x_test.toarray())
    print("BernoulliNB con TfidTransformer: ")
    bct = scoring(pred_bernoulli_tf)
    print(bct)

    multinomial = multinomial.fit(x_train_tf, y_train)
    pred_multinomial_tf = multinomial.predict(x_test.toarray())
    print("MultinomialNB con TfidTransformer: ")
    mct = scoring(pred_multinomial_tf)
    print(mct)
    print('Incremento di BernoulliNB con TfidTransformer : ', (bct-bst)*100, '%')
    print('Incremento di MultinomialNB con TfidTransformer: ', (mct-mst)*100, '%')
