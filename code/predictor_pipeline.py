# Runs a basic LinearSVC preddictor based on the properties of the model


import numpy as np
import pandas as pd
import spacy
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

def modelCreator(text, personality, pipeline):
    X = text
    y = personality
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)

    text_clf = pipeline.fit(X_train, y_train)
    predictions = text_clf.predict(X_test)
    print(metrics.accuracy_score(y_test, predictions))

    return text_clf

def main():
    essays = pd.read_csv('essays.csv',encoding='cp1252')
    essays['cTRAITS'] = essays['cEXT'] + essays['cNEU'] + essays['cAGR'] + essays['cCON'] + essays['cOPN']
    #print(essays['cEXT'].value_counts())

    # X = essays['TEXT']
    # y = essays['cEXT']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)

    # count_vect = CountVectorizer()
    # print(X)
    # X_train_counts = count_vect.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # vectorizer = TfidfVectorizer()
    # X_train_tfidf = vectorizer.fit_transform(X_train)
    # clf = LinearSVC()
    # clf.fit(X_train_tfidf, y_train)

    # text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC())])
    # text_clf.fit(X_train, y_train)
    # predictions = text_clf.predict(X_test)
    # print(metrics.accuracy_score(y_test, predictions))
    # print(text_clf.predict(["I rather stay indoors with my friends"]))



    pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC())])
    concatModel = modelCreator(essays['TEXT'], essays['cTRAITS'], pipeline)
    # extModel = modelCreator(essays['TEXT'], essays['cEXT'], pipeline)
    # neuModel = modelCreator(essays['TEXT'], essays['cNEU'], pipeline)
    # agrModel = modelCreator(essays['TEXT'], essays['cAGR'], pipeline)
    # conModel = modelCreator(essays['TEXT'], essays['cCON'], pipeline)
    # opnModel = modelCreator(essays['TEXT'], essays['cOPN'], pipeline)

    prediction_text = ["I rather stay indoors with my friends"]
    prediction_dict = {}
    prediction_dict['traits'] = concatModel.predict(prediction_text)[0]
    # prediction_dict['ext'] = extModel.predict(prediction_text)[0]
    # prediction_dict['neu'] = neuModel.predict(prediction_text)[0]
    # prediction_dict['agr'] = agrModel.predict(prediction_text)[0]
    # prediction_dict['con'] = conModel.predict(prediction_text)[0]
    # prediction_dict['opn'] = opnModel.predict(prediction_text)[0]

    print(prediction_dict)

if __name__ == "__main__":
    main()