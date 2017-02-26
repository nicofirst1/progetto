import gensim

from Preprocessing import *

lst=word_polishing(list(TRAIN_DATASET["review"]))
prova2(lst,list(TRAIN_DATASET["sentiment"]))