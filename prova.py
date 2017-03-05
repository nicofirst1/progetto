from Data_analisys import plot_vector, plot_top_n_words, plot_chi2
from Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, TEST_DATASET
from Processing import string2vecTFIDF
from util import *


x_train_vec, x_test_vec, vect=string2vecTFIDF(TRAIN_DATASET_LABLED["review"],TRAIN_DATASET_UNLABLED["review"],TEST_DATASET["review"])

plot_chi2(x_train_vec,TRAIN_DATASET_LABLED["sentiment"],vect)


#
# forest_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
# SGD_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)