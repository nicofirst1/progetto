from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import TfidfTransformer

from Data_analisys import plot_top_forest_features_importance
from util import scoring, polish_tfidf_kbest

from Preprocessing import string2vecCV, string2vecTFIDF, dimensionality_reductionKB

TO_PLOT=True


def forest_classifier(train_set_labled,train_set_unlabled,test_set):

    #tratto i dati
    xtrain_vec, xtest_vec,ytrain, vect=polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set)

    print("inizio classificazione......")

    # inizzializzo il classificatore e inizio il fittaggio
    forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=1, criterion="entropy")
    forest = forest.fit(xtrain_vec, ytrain)

    if(TO_PLOT):
        features = vect.inverse_transform(xtrain_vec)
        plot_top_forest_features_importance(forest,features,20)


    # adesso posso provare a fare la predizione
    pred_forest = forest.predict(xtest_vec)

    forest_score = scoring(pred_forest)
    print(forest_score)

def SGD_classifier(train_set_labled,train_set_unlabled,test_set):


    xtrain_vec, xtest_vec,ytrain=polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set)

    print("inizio classificazione......")


    #inizzializzo il classificatore e inizio il fittaggio
    sgd=SGDClassifier(verbose=1,n_jobs=-1,loss="modified_huber",random_state=4,n_iter=10)
    sgd.fit(xtrain_vec,ytrain)

    print("inizio predizione......")

    # adesso possoa fare la predizione
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
