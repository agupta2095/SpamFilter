import os
import glob
import numpy as np
import email
import pickle
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from sklearn.model_selection import train_test_split
import re
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction._stop_words import  ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

path = 'mail_data/'
f = open("x_train.txt", "w")
fxt = open("x_test.txt", "w")
easy_ham_paths = glob.glob(path+'easy_ham/*')
easy_ham_2_paths = glob.glob(path+'easy_ham_2/*')
hard_ham_paths = glob.glob(path+'hard_ham/*')
spam_paths = glob.glob(path+'spam/*')
spam_2_paths = glob.glob(path+'spam_2/*')


def get_email_content(path):
    try:
        file = open(path, encoding='latin1')
        try:
            msg = email.message_from_file(file)
            # iterate over email parts
            for part in msg.walk():
                # extract content type of email
                if part.get_content_type() == 'text/plain':
                    # get the email body
                    return part.get_payload()
        except Exception as ex:
            print(ex)
    except:
       print("File not found")


def get_all_email_contents(email_paths):
    email_contents = [get_email_content(p) for p in email_paths]
    return email_contents

def remove_hyperlink(word):
    return re.sub(r"http\S+", "", word)

def to_lower(word):
    return word.lower()

def remove_number(word):
    return re.sub(r'\d+', "", word)

def remove_whitespace(word):
    return word.strip()

def replace_newline(word):
    return re.sub(r'\n','', word)

def remove_punctuation(word):
    return re.sub(r'[^\w\s]', '', word)

def cleanup_sentence(sentence):
    if sentence == None:
        return sentence
    if type(sentence) != str:
        return sentence

    sentence = remove_punctuation(sentence)
    sentence = remove_number(sentence)
    sentence = remove_hyperlink(sentence)
    sentence = remove_whitespace(sentence)
    sentence = to_lower(sentence)
    return sentence

# Stop words are such that which doesn't hold much value
# Such as conjuctors, determinants to know the tone/context of textual data set
def remove_stop_words(sen):
    result = [i for i in sen if i not in ENGLISH_STOP_WORDS]
    return result

# Word stemming is removing end and beginning from words which are common prefixes/suffixes such as ing, es, etc
# Get words with same meaning irrespective of verb forms
# Not very efficient, because say word Ring -> after stemming becomes R, so works on rules
# But easiest method, so works
def word_stemmer(sen):
    return [stemmer.stem(s) for s in sen]

# Lemmatization: Reducing the word to their root or dictionary form
def word_lemmatizer(sen):
    return [lemmatizer.lemmatize(s) for s in sen]

def tokenize(sen):
    if sen is None:
        return sen
    remove_stop_words(sen)
    word_stemmer(sen)
    word_lemmatizer(sen)
    return sen

if __name__ == '__main__':
    ham_paths = [easy_ham_2_paths, easy_ham_paths, hard_ham_paths]
    spam_paths = [spam_2_paths, spam_paths]
    # This is a 2 Dimensional array with 3 rows and 2 columns
    # Split the data from each folder into two parts, one is for training and one is for testing
    ham_sample = np.array([train_test_split(o, test_size=0.25, random_state=3) for o in ham_paths], dtype=object)
    #print(ham_sample)
    #print(ham_sample.shape)
    ham_train = np.array([])
    ham_test = np.array([])
    for o in ham_sample:
        ham_test = np.concatenate((ham_test, o[1]), axis=0)
        ham_train = np.concatenate((ham_train, o[0]), axis=0)
    print(ham_train.shape)
    print(ham_test.shape)

    spam_sample = np.array([train_test_split(o, test_size=0.25, random_state=3) for o in spam_paths], dtype=object)
    spam_train = np.array([])
    spam_test = np.array([])

    for p in spam_sample:
        spam_train = np.concatenate((spam_train,p[0]), axis=0)
        spam_test = np.concatenate((spam_test, p[1]), axis=0)

    print(spam_test.shape)
    print(spam_train.shape)

    # Create label for spam mails of same size as spam_train and spam_test
    spam_train_label = [1]*spam_train.shape[0]
    spam_test_label = [1]*spam_test.shape[0]
    ham_train_label = [0]*ham_train.shape[0]
    ham_test_label = [0]*ham_test.shape[0]

    # merge the spam and ham mail for test and train data
    X_test = np.concatenate((spam_test, ham_test))
    Y_test = np.concatenate((spam_test_label, ham_test_label))

    X_train = np.concatenate((spam_train, ham_train))
    Y_train = np.concatenate((spam_train_label, ham_train_label))

    # Randomly generate a permutation of the size of X_train and X_test to the shuffle indexes of array X_train and X_test
    train_shuffled = np.random.permutation(np.arange(0, X_train.shape[0]))
    test_shuffled = np.random.permutation(np.arange(0, X_test.shape[0]))

    X_train = X_train[train_shuffled]
    X_test = X_test[test_shuffled]

    Y_train = Y_train[train_shuffled]
    Y_test = Y_test[test_shuffled]

    # Now for each mail get its body
    X_train = get_all_email_contents(X_train)
    X_test = get_all_email_contents(X_test)

    print (len(X_train))
    print(len(X_test))
    print(len(Y_train))
    print(len(Y_test))

    #Text Cleaning

    X_train = [cleanup_sentence(sen) for sen in X_train]
    X_test = [cleanup_sentence(sen) for sen in X_test]

    '''for i, o in enumerate(X_test):
        if o is None:
            continue
        fxt.write("\n-------" + str(i) + "-------\n")
        fxt.write(o)
    for i, o in enumerate(X_train):
        if o is None:
            continue
        f.write("\n-------" + str(i) +"-------\n")
        f.write(o)'''
    #Tokenization and Lemmanization
    # Tokenization means breaking sentences into words,

    f.write("AFTER TOKENIZATION\n")
    X_token = []
    for o in X_train:
        if o is None:
            X_token.append(o)
            continue
        if type(o) is not str:
            continue
        X_token.append(word_tokenize(o))
    X_train = X_token
    '''
    for i, o in enumerate(X_token):
        if o is None:
            continue
        f.write("\n-------" + str(i) + "-------\n")
        for word in o:
            f.write(word + " ")
        f.write("\n")
    '''
    X_test_token = []
    for o in X_test:
        if o is None:
            X_test_token.append(o)
            continue
        if type(o) is not str:
            continue
        words = word_tokenize(o)
        #for w in words:
        #    fxt.write(w +" ")
        #fxt.write("\n")
        X_test_token.append(words)

    X_test = X_test_token

    X_train = [tokenize(sen) for sen in X_train]
    X_test = [tokenize(sen) for sen in X_test]

    for i, o in enumerate(X_train):
        if o is None:
            continue
        f.write("\n-------" + str(i) + "-------\n")
        for word in o:
            f.write(word + " ")
        f.write("\n")

    for i, o in enumerate(X_test_token):
        if o is None:
            continue
        fxt.write("\n-------" + str(i) + "-------\n")
        for word in o:
            fxt.write(word + " ")
        fxt.write("\n")


    for i, s in enumerate(X_train):
        if s is not None:
            X_train[i] = " ".join(s)

    for i, s in enumerate(X_test):
        if s is not None:
            X_test[i] = " ".join(s)

    f1 = open("Train_X.pkl", "wb")
    f2 = open("Train_Y.pkl", "wb")
    f3 = open("Test_X.pkl", "wb")
    f4 = open("Test_Y.pkl", "wb")
    print(len(X_train))
    print(len(X_test))
    print(len(Y_train))
    print(len(Y_test))
    pickle.dump(X_train, f1)
    pickle.dump(Y_train, f2)
    pickle.dump(X_test, f3)
    pickle.dump(Y_test, f4)
    f.close()
    fxt.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()