from Data_analisys import *
from Processing import *

#pricipal_component(TRAIN_DATASET)

forest_grid(TRAIN_DATASET, TEST_DATASET)
dictionary=cluster_dict_syn()
semantic_cluster_assignment(dictionary)