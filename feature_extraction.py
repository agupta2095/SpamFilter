# The words need to be encoded as integers or floating point values for use as input to machine learning algorithms

# The scikit library offers easy ways for both tokenization and feature extraction of your text data

from sklearn.feature_extraction.text import TfidfVectorizer
from visualization import read_pickle
import numpy as np

X_train = np.array(read_pickle("Train_X.pkl"), dtype=object)
Y_train = np.array(read_pickle("Train_Y.pkl"))
X_test = np.array(read_pickle("Test_X.pkl"), dtype=object)
Y_test = np.array(read_pickle("Test_Y.pkl"))

def convert_to_feature(raw_tokenize_data):
    raw_sentences = [' '.join(o) for o in raw_tokenize_data]
    return vectorizer.transform(raw_sentences)

if __name__ == '__main__':
    X_train = [s.split(" ") for s in X_train]
    X_test = [s.split(" ") for s in X_test]
    vectorizer = TfidfVectorizer()
    raw_sentences = [' '.join(o) for o in X_train]
    vectorizer.fit(raw_sentences)

    x_train_features = convert_to_feature(X_train)
    x_test_features = convert_to_feature(X_test)
