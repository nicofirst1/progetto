from os import system

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from src.Preprocessing import TEST_DATASET, string2vecTFIDF, dimensionality_reductionKB


# Todo: cross validation per tutning dei parametri


def cross_validation_score(model, x):
    scores = cross_val_score(model, x, y=TEST_DATASET["sentiment"])
    print(scores)

def save_to_csv(pred):

    # saving predicton for sending it to kaggle
    dict={"id":TEST_DATASET["id"],"sentiment":pred}
    output=pd.DataFrame(data=dict)
    output.to_csv("res.csv", index=False, quoting=3)

def scoring(prediction, true,what):

    #using mean square error for calculating accurancy
    pred = str(np.mean(prediction == true) * 100)
    #printing score
    print("Score for "+what+" is: " + pred)
    string = "Score is " + pred + " percent"
    system('say ' + string)


def polish_tfidf_kbest(train_set_labled, train_set_unlabled, test_set):
    # splitting  train test
    xtrainL = train_set_labled["review"]
    xtrainU = train_set_unlabled["review"]
    xtest = test_set["review"]
    ytrain = train_set_labled["sentiment"]

    print("Starting trasformation from string to vector...")

    # transforming to vector
    xtrain_vec, xtest_vec, vect = string2vecTFIDF(xtrainL, xtrainU, xtest)

    feature_names = vect.get_feature_names()

    print("Executing chi2 test...")

    reduced_xtrain_vec, reduced_xtest_vec = dimensionality_reductionKB(xtrain_vec, ytrain, xtest_vec,feature_names)
    return reduced_xtrain_vec, reduced_xtest_vec, ytrain, feature_names


def svm_grid(xtrain, ytrain):
    svm = RandomForestClassifier(n_jobs=-1, verbose=1, criterion="entropy")

    param = {"n_estimators": [50, 100, 150, 200, 250, 300]}

    grid = GridSearchCV(svm, param, n_jobs=-1, verbose=1, error_score=-1)
    grid.fit(xtrain, ytrain)

    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)

def save_wrong_answer(pred,names,xtest):

    # n is the number of feature to print
    n=3


    # saving true values
    true=TEST_DATASET["sentiment"]
    wrong=[]

    idx=0
    index=[]

    # for every cuple (predicted, true value)
    for i,j in zip(pred,true):
        # if values are not the same
        if (i!=j):
            # adding i-th review to the list plus incorrect prediction
            wrong.append((idx,TEST_DATASET["review"].iget(idx),i))
            # saving indices of wrong prediction
            index.append(idx)
        idx+=1


    # saving file with wrong prediction for further analysis
    with open("/Users/nicolo/PycharmProjects/progetto/wrong.txt","w") as file:
        file.write('\n\n'.join('%d) %s \t Incorrect= %s' % x for x in wrong))

    print("wrong file saved")

    # taking all missclassified reviews
    samples=[]
    for elem in index:
        samples.append(xtest[elem])

    # for evrey misscalssified review, take corresponding features values
    correct=[]
    for line in samples:
        # linking features with words
        line=[y for y in zip(line,names)]
        # taking top-n features
        line=sorted(line,key= lambda tup: tup[0])[-n:]
        # appending to list
        correct.append(line)


    anlayzer=[]
    idx=0
    # analyzer have this form (index, (value, word))
    for elem in index:
        anlayzer.append((elem,correct[idx]))
        idx+=1

        # save file
    with open("/Users/nicolo/PycharmProjects/progetto/wrong_data.txt", "w") as file:
        file.write('\n\n'.join('%d) %s' % x for x in anlayzer))

    print("wrong_data file created")



