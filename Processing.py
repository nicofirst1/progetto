import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV

from Data_analisys import scoring

from Preprocessing import  TEST_DATASET, string2vecCV



def forestGrid(train_dataset, test_dataset):
    # prima di tutto salvo le y_train
    y_train = train_dataset["sentiment"]

    # poi mando il train datasaet nel preprocessing usando il CountVectorizer
    x_train, x_test = string2vecCV(train_dataset["review"], x_test_str=test_dataset["review"])

    param_grid={'n_estimators':[100,300,500,1000],'criterion':['gini','entropy'],'min_samples_split':[2,4,16,32,64],
                 'min_samples_leaf':[1,5,10],'bootstrap':[True,False],'oob_score':[True,False]}

    grid=GridSearchCV(RandomForestClassifier,param_grid,n_jobs=-1,verbose=True)
    grid.fit(x_train,y_train)
    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)


def forest(train_dataset, test_dataset):

    #prima di tutto salvo le y_train
    y_train=train_dataset["sentiment"]

    #poi mando il train datasaet nel preprocessing usando il CountVectorizer
    x_train, x_test =string2vecCV(train_dataset["review"],x_test_str=test_dataset["review"])

    #inizzializzo l'estimatore e inizio il fittaggio
    forest = RandomForestClassifier(n_estimators=300,n_jobs=-1, verbose=1, criterion="entropy")
    forest=forest.fit(x_train,y_train)



    #adesso posso provare a fare la predizione
    pred=forest.predict(x_test)

    # #salvo i risultati in un dataframe e li trasformo in csv
    # to_save=pd.DataFrame(data={"id":TEST_DATASET["id"],"sentiment":pred})
    # to_save.to_csv("pred.csv", index=False, quoting=3 )

    scoring(pred)
