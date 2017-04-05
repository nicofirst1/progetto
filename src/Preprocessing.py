import os
import pandas as pd
import time
import numpy as np
from nltk.stem import SnowballStemmer
from gensim.parsing.preprocessing import  strip_punctuation2, strip_non_alphanum, strip_tags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from os import system
from nltk.corpus import stopwords
from multiprocessing import Pool

# Todo: stemming, stopwords


# saving for evry dataset

TEST_PATH = os.getcwd() + "/datasets/testDataLabeled.tsv"
TRAIN_PATH_LABLED = os.getcwd() + "/datasets/labeledTrainData.tsv"
TRAIN_PATH_UNLABLED = os.getcwd() + "/datasets/unlabeledTrainData.tsv"

TEST_DATASET = pd.read_csv(TEST_PATH, header=0, sep="\t", quoting=3)
del TEST_DATASET["Unnamed: 0"]
TRAIN_DATASET_LABLED = pd.read_csv(TRAIN_PATH_LABLED, header=0, sep="\t", quoting=3)
TRAIN_DATASET_UNLABLED = pd.read_csv(TRAIN_PATH_UNLABLED, header=0, sep="\t", quoting=3)



def spot_differences(old, new, what, words):

    i=0
    j=0
    ris=""
    print("difference for: "+what)
    for old_row, new_row in zip(old,new):
        for old_cell, new_cell in zip(old_row,new_row):
            if(old_cell!= new_cell):
                ris+="difference for cell ["+str(i)+", "+str(j)+", "+words[j]+" ]\t"+str(old_cell)+" -> "+str(new_cell)+"\n"

            j+=1

        i+=1
    with open("difference_"+what+".txt","w") as file:
        file.write(ris)



def sentences_polishing(words_lst, what, deep_polishing=True, essential=False):

    # calculating char numers for entire review list
    lst_len_start = sum(len(s) for s in words_lst)
    print("Cleaning for list with " + str(lst_len_start) + " chars, for " + what)

    # deleting html tags
    words_lst = [strip_tags(x) for x in words_lst]

    if not essential:

        # deleting punctuation
        words_lst = [strip_punctuation2(x) for x in words_lst]

        # deleting non alphanumeric chars
        words_lst = [strip_non_alphanum(x) for x in words_lst]

    if deep_polishing:
        # Initializing pool for multiprocessing
        pool = Pool(processes=10)

        # for every review, apply function and save result
        words_lst = pool.map(stemming_stopWords, words_lst)
        pool.close()
        pool.join()

    # deleting empty reviews
    words_lst = [x for x in words_lst if x]

    # recalculating list char and printing results
    lst_len_end = sum(len(s) for s in words_lst)
    cleaned = lst_len_start - lst_len_end
    print("Deleted " + str(cleaned) + " (" + str(int(cleaned / lst_len_start * 100)) + "%) chars, for " + what + "\n")

    return words_lst


def stemming_stopWords(review):
    # Initializating stemmatiser
    stemmer = SnowballStemmer("english")

    # splitting into words
    words_list = review.split()

    # deleting words with less than 3 chars
    words_list = [x for x in words_list if len(x) > 2]

    # removing stop words
    words_list = [word for word in words_list if word not in stopwords.words('english')]

    # applying stemmatizer
    words_list = [stemmer.stem(x) for x in words_list]

    # removing numbers
    words_list = [word for word in words_list if word.isalpha()]

    return " ".join(words_list)


def string2vecTFIDF(x_train_str_labled, x_train_str_unlabled, x_test_str):
    print("Cleaning for train dataset\n")
    start = time.time()

    # cleaning every dataset
    clean_x_trainL = sentences_polishing(list(x_train_str_labled), "XtrainLabled")
    clean_x_trainU = sentences_polishing(list(x_train_str_unlabled), "XTrainUnlabled")
    clean_x_test = sentences_polishing(list(x_test_str), "XTest")

    #inizializing tfidfVectorizer
    vect = TfidfVectorizer(min_df=5, max_df=0.80, sublinear_tf=True, max_features=85000,
                           strip_accents='unicode', token_pattern=r'\w{1,}',
                           ngram_range=(1, 2))

    # fitting train labled and unlabled into tfidf
    vect = vect.fit(clean_x_trainL + clean_x_trainU)
    # transforming train dataset
    x_train_vec = vect.transform(clean_x_trainL).toarray()
    print("train dataset trasformato, creato dizionario con " + str(len(vect.vocabulary_)) + " parole")

    # transforming test dataset
    x_test_vec = vect.transform(clean_x_test).toarray()

    # taking time and printing result
    end = time.time()
    tot = end - start

    print("Cleaning and transformation for  test dataset done\nTotal time: " + str(int(tot / 60)) + "' " + str(
        int(tot % 60)) + "''\n")
    system('say "Cleaning for dataset done"')

    return x_train_vec, x_test_vec, vect


def dimensionality_reductionKB(xtrain, ytrain, xtest,names, percentage=0.85):
    print("Starting reduction...")
    start = time.time()

    # select how many features to leave
    k = int(len(xtrain) * percentage)
    kbest = SelectKBest(chi2, k=k)
    new_xtrain = kbest.fit_transform(xtrain, ytrain)
    print("xtrain reduced!")
    new_xtest = kbest.transform(xtest)
    end = time.time()
    tot = end - start

    spot_differences(xtrain,new_xtrain,"xtrain for chi2",names)

    print("xtest reduced!\nTotal time: " + str(int(tot / 60)) + "' " + str(int(tot % 60)) + "''\n")
    system('say "Ki square test completed"')

    return new_xtrain, new_xtest
