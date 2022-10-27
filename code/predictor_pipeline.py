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
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn import metrics

def modelCreator(text, personality, pipeline, name):
    X = text.astype(str)
    y = personality
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
    text_clf = pipeline.fit(X_train, y_train)
    predictions = text_clf.predict(X_test)
    print(f'{name}, {text.name}: {metrics.accuracy_score(y_test, predictions)}')

    return text_clf

def accuracy(text, personality, pipeline, name):
    X = text.astype(str)
    y = personality
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
    text_clf = pipeline.fit(X_train, y_train)

    predictions = text_clf.predict(X_test)

    return metrics.accuracy_score(y_test, predictions)

def accuracy_finder():
    essays = pd.read_csv('final_v2.csv',encoding='cp1252')
    pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf',SVC(kernel='poly'))])
    all_features = essays.columns[7:37].tolist()
    print(all_features)
    res = []
    for feat in all_features:
        print(feat)
        if feat in ["most_freq_word_length", "num_pos_coord_conj", "num_pos_det"]:
            continue

        extAccuracy = accuracy(essays[feat], essays['cEXT'], pipeline, 'Extraversion')
        neuAccuracy = accuracy(essays[feat], essays['cNEU'], pipeline, 'Neuroticism')
        agrAccuracy = accuracy(essays[feat], essays['cAGR'], pipeline, 'Agreeableness')
        conAccuracy = accuracy(essays[feat], essays['cCON'], pipeline, 'Conscientiousness')
        opnAccuracy = accuracy(essays[feat], essays['cOPN'], pipeline, 'Openness')

        res.append([feat,extAccuracy,neuAccuracy,agrAccuracy,conAccuracy,opnAccuracy])

    df = pd.DataFrame(res, columns = ['Feature', 'Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness'])
    df.to_csv('accuracy.csv', sep=',', encoding='utf-8', index = False) 
    

def main():
    # essays = pd.read_csv('final_v2.csv',encoding='cp1252')
    # essays['cTRAITS'] = essays['cEXT'] + essays['cNEU'] + essays['cAGR'] + essays['cCON'] + essays['cOPN']
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



    # pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC())])
    # pipeline = Pipeline([('clf',LinearSVC())])
    # concatModel = modelCreator(essays['TEXT'], essays['cTRAITS'], pipeline)
    # print(essays['num_sentences'])

    #Testing Traits
    # Grab the necessary features
    # sig_feat = pd.read_csv('./analysis_data/significant_features_altered.csv',encoding='cp1252')
    # for feat in sig_feat.iloc[0]['signf_features'].split(','):
    #     extModel = modelCreator(essays[feat], essays['cEXT'], pipeline, 'Extraversion')

    # for feat in sig_feat.iloc[1]['signf_features'].split(','):
    #     neuModel = modelCreator(essays[feat], essays['cNEU'], pipeline, 'Neuroticism')

    # for feat in sig_feat.iloc[2]['signf_features'].split(','):
    #     agrModel = modelCreator(essays[feat], essays['cAGR'], pipeline, 'Agreeableness')
    
    # for feat in sig_feat.iloc[3]['signf_features'].split(','):
    #     conModel = modelCreator(essays[feat], essays['cCON'], pipeline, 'Conscientiousness')

    # for feat in sig_feat.iloc[4]['signf_features'].split(','):
    #     opnModel = modelCreator(essays[feat], essays['cOPN'], pipeline, 'Openness')


    # all_features = essays.columns[6:36].tolist()
    # for feat in all_features:
    #     extModel = modelCreator(essays[feat], essays['cEXT'], pipeline, 'Extraversion')

    # for feat in all_features:
    #     neuModel = modelCreator(essays[feat], essays['cNEU'], pipeline, 'Neuroticism')
    
    # for feat in all_features:
    #     agrModel = modelCreator(essays[feat], essays['cAGR'], pipeline, 'Agreeableness')

    # for feat in all_features:
    #     conModel = modelCreator(essays[feat], essays['cCON'], pipeline, 'Conscientiousness')

    # for feat in all_features:
    #     opnModel = modelCreator(essays[feat], essays['cOPN'], pipeline, 'Openness')

    # neuModel = modelCreator(essays['most_freq_word_length'], essays['cNEU'], pipeline, 'Neuroticism')

    # extModel = modelCreator(essays['LIX_readability'], essays['cEXT'], pipeline, 'Extraversion')
    # neuModel = modelCreator(essays['num_sentences'], essays['cNEU'], pipeline, 'Neuroticism')
    # agrModel = modelCreator(essays['hapax_legomena'], essays['cAGR'], pipeline, 'Agreeableness')
    # conModel = modelCreator(essays['dale_chall_readability'], essays['cCON'], pipeline, 'Conscientiousness')
    # opnModel = modelCreator(essays['num_diff_word_nstop'], essays['cOPN'], pipeline, 'Openness')

    # prediction_text = ["I rather stay indoors with my friends"]
    # prediction_dict = {}
    # # prediction_dict['traits'] = concatModel.predict(prediction_text)[0]
    # prediction_dict['ext'] = extModel.predict(prediction_text)[0]
    # prediction_dict['neu'] = neuModel.predict(prediction_text)[0]
    # prediction_dict['agr'] = agrModel.predict(prediction_text)[0]
    # prediction_dict['con'] = conModel.predict(prediction_text)[0]
    # prediction_dict['opn'] = opnModel.predict(prediction_text)[0]

    # print(prediction_dict)

    accuracy_finder()

if __name__ == "__main__":
    main()