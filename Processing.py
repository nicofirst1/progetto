from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import *

from Data_analisys import plot_SGD_vect, plot_forest_vect, plot_SGD_decision
from Preprocessing import string2vecCV
from util import scoring, polish_tfidf_kbest, cross_validation_score

TO_PLOT=True



def forest_classifier(train_set_labled,train_set_unlabled,test_set):

    #tratto i dati
    xtrain_vec, xtest_vec,ytrain=polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set)

    print("inizio classificazione......")

    # inizzializzo il classificatore e inizio il fittaggio
    forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=1, criterion="entropy")
    forest = forest.fit(xtrain_vec, ytrain)

    if (TO_PLOT):
        plot_forest_vect(forest)

    # cross_validation_score(forest,xtest_vec)
    # return

    # oob_error = 1 - forest.oob_score_
    # print("oob_error: "+str(oob_error))

    # adesso posso provare a fare la predizione
    pred_forest = forest.predict(xtest_vec)

    scoring(pred_forest)

def SGD_classifier(train_set_labled,train_set_unlabled,test_set):


    xtrain_vec, xtest_vec,ytrain=polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set)

    print("inizio classificazione......")

    #inizzializzo il classificatore e inizio il fittaggio
    sgd=SGDClassifier(verbose=1,n_jobs=-1,loss="modified_huber",random_state=4,n_iter=10,
                      shuffle=True)

    sgd.fit(xtrain_vec,ytrain)

    if(TO_PLOT):
        plot_SGD_decision(sgd,xtrain_vec,ytrain)
        #plot_SGD_vect(sgd)


    print("inizio predizione......")

    # adesso possoa fare la predizione
    pred_sgd = sgd.predict(xtest_vec)

    scoring(pred_sgd)




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
