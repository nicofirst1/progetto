import gensim

from Preprocessing import *

lst=word_polishing(list(TRAIN_DATASET["review"]))
forest(lst,list(TRAIN_DATASET["sentiment"]))