import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC, SVR

def trait_feature_accuracy(feature, trait, source, rs=42):
    X = source[feature]
    y = source[trait]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=rs)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    # df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions))
    acc_scr = metrics.accuracy_score(y_test, predictions)
    # print("Feature: {}/Trait: {}/Acc_Score: {}".format(feature,trait,acc_scr))
    return acc_scr

if __name__ == "__main__":

    fv_csv_file = 'final_v2.csv' # Raw Final Data
    zsc_csv_file = 'z_score_final_v2.csv' # Z Score data

    fv_feature_df = pd.read_csv(fv_csv_file, index_col = 0)
    zsc_feature_df = pd.read_csv(zsc_csv_file, index_col = 0)

    all_features = fv_feature_df.columns[6:36].tolist()
    all_personality = ['cEXT','cNEU', 'cAGR', 'cCON', 'cOPN']

    results = []
    
    # for trait in all_personality:
    #     for feature in all_features:
    #             accuracy = trait_feature_accuracy(feature, trait, fv_feature_df, 88)
    #             results.append([feature, trait, accuracy])

    anova_features = pd.read_csv('./analysis_data/trait_sig_feat_anova.csv', index_col = 0)
    sig_feature_ttest = pd.read_csv('./analysis_data/significant_features.csv', index_col = 0)

    for trait in all_personality:
        print(trait)
        trait_sig_features = anova_features.loc[trait]['sig_features'].split(", ")
        # trait_sig_features = sig_feature_ttest.loc['{}_t_stat'.format(trait)]['signf_features'].split(",")
        # print(trait_sig_features)
        accuracy = trait_feature_accuracy(trait_sig_features, trait, fv_feature_df, 808)
        results.append([trait, accuracy])

    res_df = pd.DataFrame(results, columns = ['trait','accuracy'])
    res_df.to_csv('./analysis_data/acc_Sig_Anova_LR_3.csv', sep=',', encoding='utf-8', index = False) 