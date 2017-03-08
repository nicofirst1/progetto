from Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED
from Processing import SGD_classifier, forest_classifier
from util import *


# x_train_vec, x_test_vec, vect=string2vecTFIDF(TRAIN_DATASET_LABLED["review"],TRAIN_DATASET_UNLABLED["review"],TEST_DATASET["review"])
#
# plot_chi2_vect(x_train_vec,TRAIN_DATASET_LABLED["sentiment"])
#

#
forest_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
# SGD_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
# #naive_bayes(TRAIN_DATASET_LABLED,TEST_DATASET)


# x_train_vec, x_test_vec, vect=string2vecTFIDF(TRAIN_DATASET_LABLED["review"],TRAIN_DATASET_UNLABLED["review"],TEST_DATASET["review"])
# plot_SGD_top(x_train_vec,TRAIN_DATASET_LABLED["sentiment"],x_test_vec,vect)