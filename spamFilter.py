# Importing the dependencies

# TfidfVectorizer will use to convert the text into feature vectors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data collection and Pre-processing
# Load the csv file to a pandas Data Frame
# Replace the null values with null strings
def read_data():
    email_data = pd.read_csv('mail_data.csv')
    email_data.where((pd.notnull(email_data)), '', inplace=True)

    # printing data frame
    #print(email_data)
    #print(email_data.head())

    # Checking the number of rows and columns in data frame
    # 5572 rows X 2 columns
    # Columns are Category and Message
    #print(email_data.shape)

    # Label encoding
    # Label Spam mail as 0 and Ham mail as 1
    email_data.loc[email_data['Category'] == 'spam', 'Category'] = 0
    email_data.loc[email_data['Category'] == 'ham', 'Category'] = 1
    #print(email_data)

    # Separating the data as text and labels

    X = email_data['Message']
    Y = email_data['Category']
    #print(X)
    #print(Y)

    # Split X and Y into training data and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    # Printing the test data
    print(X_test.shape)
    print(Y_test.shape)
    # Printing the training data
    print(X_train.shape)
    print(Y_train.shape)

    # Transform the messages text data to feature vectors that can be used as input to the Logistic Regression models
    # Feature extraction: Mapping from textual data to real valued vectors
    # TF-IDF = Term Frequency-Inverse Document Frequency : To count the number of times each word appears in a document
    # Vectorizer doesn't understand the context of paragraph
    # Term Frequency = n(Number of times term T in the document)/N(Number of terms in the document)
    # Inverse Document Frequency (log(N(number of documents)/n(number of documents containing a word T)))
    # IDF of rare word is high, common word is low
    def feature_extraction():
        # min_df = minimum score given to words, ignore words having score lower than min_df
        features = TfidfVectorizer(min_df=1, stop_words='english', lowercase="True")
        X_train_features = features.fit_transform(X_train)
        X_test_features = features.transform(X_test)
        #print(X_train_features)
        #print(X_test_features)

        # convert Y_test and Y_train labels values as integers
        Y_test_int = Y_test.astype('int')
        Y_train_int = Y_train.astype('int')

        # Training the machine learning Model
        # Logistic Regression Model
        model = LogisticRegression()
        # Training the Logistic Regression model with training data
        model.fit(X_train_features, Y_train_int)

        # Evaluating the trained model
        # Prediction on test data

        prediction_on_test_data = model.predict(X_test_features)
        accuracy_on_training_data = accuracy_score(Y_test_int, prediction_on_test_data)

        print('Accuracy on test data :', accuracy_on_training_data)

        # Building a predictive System
        input_mail = [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entr..."]
        input_mail2= ["Even my brother is not like to speak with me. They treat me like aids patent."]

        input_mail3=["As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertu.."]

        input_mail4 =["WINNER!! As a valued network customer you have been selected to receivea �900 prize reward! To claim..."]

        input_mail_features = features.transform(input_mail4)

        # Predicted right for the above two mails
        if( model.predict(input_mail_features) == 0) :
            print("SPAM !!")
        else:
            print("No Spam, Go Ahead and read!")
    feature_extraction()


if __name__ == '__main__':
    read_data()



