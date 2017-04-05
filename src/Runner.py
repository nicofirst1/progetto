from Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED

from src.Processing import SVC_classifier

## TRATTAMENTO DATI
# x_train_vec, x_test_vec, vect=string2vecTFIDF(TRAIN_DATASET_LABLED["review"],TRAIN_DATASET_UNLABLED["review"],TEST_DATASET["review"])
#x_train_vec, x_test_vec, ytrain,xval_vec, names = polish_tfidf_kbest(TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, TEST_DATASET,VALIDATION_SET)

## PLOTS
# plot_chi2_vect(x_train_vec,TRAIN_DATASET_LABLED["sentiment"])
# plot_chi2_top(x_train_vec,TRAIN_DATASET_LABLED["sentiment"],vect)
# plot_top_n_words(vect,10)

## CLASSIFICAZIONE
#forest_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
SVC_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)


## PLOTTING LEARNING RATE
#forest = RandomForestClassifier(n_estimators=250, n_jobs=-1, verbose=1, criterion="entropy")
#svc=LinearSVC(verbose=True, penalty="l2", loss="hinge")

#plot_learning_curve(forest,"Forest learning curve",x_train_vec,ytrain)
#plot_learning_curve(svc,"SVC learning curve",x_train_vec,ytrain)

#svm_grid(x_test_vec,TRAIN_DATASET_LABLED["sentiment"])

## PROVA
#multiple_classifier()


from os import system

system('say "Esecuzione terminata"')
system('say "Esecuzione terminata"')

