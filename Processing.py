import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from Data_analisys import scoring

from Preprocessing import  TEST_DATASET, string2vecCV


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
