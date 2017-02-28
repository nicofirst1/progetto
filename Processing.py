import pandas as pd
import numpy as np
from gensim.parsing import strip_punctuation2
from gensim.parsing import strip_tags
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV

from Data_analisys import scoring

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



@DeprecationWarning
def forestGrid(train_dataset, test_dataset):
    # prima di tutto salvo le y_train
    y_train = train_dataset["sentiment"]

    # poi mando il train datasaet nel preprocessing usando il CountVectorizer
    x_train, x_test = string2vecCV(train_dataset["review"], x_test_str=test_dataset["review"],max_features=5000)

    param_grid={'n_estimators':[100,300,500,1000],'criterion':['entropy']}

    grid=GridSearchCV(RandomForestClassifier(),param_grid,n_jobs=-1,verbose=True)
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
