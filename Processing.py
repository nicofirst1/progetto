import pandas as pd
import numpy as np
from gensim.parsing import strip_punctuation2
from gensim.parsing import strip_tags
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import TfidfTransformer
from Data_analisys import scoring, time

from Preprocessing import string2vecCV, TRAIN_PATH_LABLED, string2vecTFIDF, dimensionality_reductionKB


def forest_classifierCV(train_dataset, test_dataset):

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

    forest_score = scoring(pred_forest)
    print(forest_score)

def SGD_classifier2(train_set_labled,train_set_unlabled,test_set):

    #faccio le divisioni
    xtrainL=train_set_labled["review"]
    xtrainU=train_set_unlabled["review"]
    ytrain=train_set_labled["sentiment"]
    xtest=test_set["review"]

    start=time.time()

    print("inizio trasformazione da stringa  a vettore......")

    #trasformo da stringhe a vettori
    xtrain_vec, xtest_vec=string2vecTFIDF(xtrainL,xtrainU,xtest)
    end=time.time()

    print("tempo impiegato: "+str(end-start)+" secondi\n")

    print("inizio dimensionality rediction......")
    start = time.time()

    #eseguo una ricerca delle labels migliori
    xtrain_vec,xtest_vec=dimensionality_reductionKB(xtrain_vec,ytrain,xtest_vec,percentage=0.9)
    end=time.time()
    print("tempo impiegato: "+str(end-start)+" secondi\n")



    print("inizio classificazione......")
    start = time.time()


    #inizzializzo il classificatore e inizio il fittaggio
    sgd=SGDClassifier(verbose=1,n_jobs=-1,loss="modified_huber",random_state=4,n_iter=10)

    sgd.fit(xtrain_vec,ytrain)

    #forest=sgd.fit(xtrain_vec,ytrain)
    end=time.time()
    print("tempo impiegato: "+str(end-start)+" secondi\n")


    print("inizio predizione......")

    # adesso posso provare a fare la predizione
    pred_sgd = sgd.predict(xtest_vec)

    forest_score = scoring(pred_sgd)
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
