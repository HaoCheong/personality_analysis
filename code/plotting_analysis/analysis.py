import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import scipy as sp
from scipy.stats import f_oneway
import math
import plotly.express as px
import pingouin as pg
from math import e
from sklearn.decomposition import PCA

from readability import Readability

# ===== STARTUP =====

fv_csv_file = '../final_v4.csv' # Raw Final Data
zsc_csv_file = '../z_score_final_v2.csv' # Z Score data

fv_feature_df = pd.read_csv(fv_csv_file, index_col = 0)
zsc_feature_df = pd.read_csv(zsc_csv_file, index_col = 0)

all_personality = ['cEXT','cNEU', 'cAGR', 'cCON', 'cOPN']
all_features = fv_feature_df.columns[6:36].tolist()

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

def two_col_pairplot(field1, field2, feature_df):
    sns.relplot(x=field1, y=field2, data=feature_df)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_sig_corr():
    corr_df = pd.read_csv('../analysis_data/correlation/ps_corr.csv')
    lo_sig_corr_df = corr_df[corr_df['ps_corr'] < -0.9]
    hi_sig_corr_df = corr_df[corr_df['ps_corr'] > 0.9]
    sig_corr_df = pd.concat([lo_sig_corr_df, hi_sig_corr_df])

    return sig_corr_df

# ===== EXTRACT ALL SIGNIFICANT CORRELATION =====

# Pair plot all the those of significant correlation
def pairplot_sig_correlation():
    sig_corr_df = get_sig_corr()
    for index, row in sig_corr_df.iterrows():
        two_col_pairplot(row[0], row[1], fv_feature_df)
        plt.savefig('../plots/sig_corr_pairplot/sig_corr_{}_{}'.format(row[0], row[1]))
        plt.clf()

# Get a list of features and their significant correlation
def feature_to_sig_corr_feature():
    sig_corr_df = get_sig_corr()
    sig_feats = sig_corr_df['feature_1'].values.tolist() + sig_corr_df['feature_2'].values.tolist()
    no_dup_sig_feat = list(set(sig_feats))
    all_feat = []
    for feat in no_dup_sig_feat:
    # feat = 'num_of_char'
        feat_sig_feat = []
        
        left_side = sig_corr_df[sig_corr_df['feature_1'] == feat]
        for i, row in left_side.iterrows():
            feat_sig_feat.append(row[1])

        right_side = sig_corr_df[sig_corr_df['feature_2'] == feat]
        for i, row in right_side.iterrows():
            feat_sig_feat.append(row[0])

        res = [feat,", ".join(feat_sig_feat), len(feat_sig_feat)]
        all_feat.append(res)


    feature_to_sig_feature = pd.DataFrame(all_feat, columns=['feature','sig_feature','count']).sort_values(by=['count'], ascending=False)
    feature_to_sig_feature.to_csv('../analysis_data/feat_to_sig_feat.csv', sep=',', encoding='utf-8', index = False)

# ========== Cronbach Alpha Sig features ==========
def cronbach_alpha_sig_feat():
    # sig_corr_df = get_sig_corr()

    # Generate a list
    # Pick the list of significant correlated features for each trait
    # features = ["dale_chall_readability", "automated_readability_index", "flesch_kincaid_grade_level", "flesch_reading_ease", "avg_sentence_length"]
    features = ["num_of_char", "cEXT"]
    sig_col_df = fv_feature_df[features]
    cb_a = pg.cronbach_alpha(sig_col_df, ci=.95)
    print(cb_a)

def information_gain():
    for feat in all_features:

        column = fv_feature_df[feat]
        base=None

        # print(column.head(10))

        vc = pd.Series(column).value_counts(normalize=True, sort=False)
        base = e if base is None else base
        ent = -(vc * np.log(vc)/np.log(base)).sum()
        print("{}: {}".format(feat, ent))

# IDK how PCA works 
def PCA_calc():
    pca_df = fv_feature_df[all_features]
    pca = PCA(n_components=30)
    pca.fit(pca_df)

    print(pca.components_)

def data_stats():
    all_row = []
    for feat in all_features:
        row = [feat]
        # print(fv_feature_df[feat])
        row.append(min(fv_feature_df[feat]))
        row.append(max(fv_feature_df[feat]))
        row.append(fv_feature_df[feat].mean())
        row.append(fv_feature_df[feat].median())
        row.append(fv_feature_df[feat].std())
        row.append(fv_feature_df[feat].kurtosis())
        all_row.append(row)

    statistical_df = pd.DataFrame(all_row, columns=['feature','min','max','mean','median','std', 'kurtosis'])
    statistical_df.to_csv('../analysis_data/feature_stats.csv', sep=',', encoding='utf-8', index = False)

# from readability import Readability

def sandbox():
    # basic_stat_df = fv_feature_df['feature']
    # text = "it is cold in my room the room is the freezing rain which chills my typing hands numbly i feel all of this through skin of course the skiing is always on top of the muscles but the muscle are in the yurget zone of the world i know that the skin is there but it must be cold roommate types next to me like a fearlees wombat that he is i must crack him open and fry him up like an egg an egg of a tale of this land which i live in is the way to a free market economy an economy which the uzbeks can have a say in their government because karimov allow them vote in private elected parties in the spindletop texas cynthia harrington went to that and saw the president president george bush sr was there and he was wearing a green rain coat a coat which i saw on the picture which was on their refrigerator door i knew that it was there because i saw it there next to the pills that their daughter constantly took she was addicted and i tried to stop her but she was depressed and i tried but you can't always help others that don't want to be helped especially those of the race of the unwilling the unwilling whose bones shall be used to pave the way to valhalla which is the greater good of the viking society which i will use to fight off the endless hordes in my brain the viking are outnumbering in my spaceship which i use the toothpaste goo food i eat it and shoot out transformer feces into my face of the po po man i will see the super lucky cat on the last date that i went on in beaumont it will be here that i go and see everything that ever was and everything that will ever be because that goes towards the greater good of mankind and i will see the sphinx before the phoenix rises out of the creamed corn of the children man man yogurt blossom in the cafeteria like bomb shells exploding in the darkness of siagon i will see them and laugh like the little devil that i am the yogurt man of my brain laughs with them and he laughs at the absurdity of it all at the absurdity of the caribbean chick which is in my class but does not see the truth which is me in the flesh and doe not see the truth which is me in the flesh of life i want to know everything i want to see everyone i want to fuck everyone in the world i want to do something that matters but the things that matter don't matter anymore my yogurt blossom repairman i thin k that i like you yogurt blossom time stream of consciousness test we are one in the same you and i watching as other s write time ticking endlessly away and our scroll bars move down ever so slightly and we presses the finish button and everyone goes wow that test really sucked and the professor get all the money and i say to you blessed are the meek they shall inherit the turf of the astrodome where i went when i was twelve to eat a dome dog and watch that team play my dad gave us peanuts to eat and i ate them and then i beat some poor bastard in the head with them and then i laughed because it was very funny and i laughed and it was funny but a whitney brown was not funny instead he was a stupid son of a bitch and not very funny at all except his face and his small groin which was funny funny funny hahahahah this must sound crazy to you i know because like me everything is crazy i am the crazy man bob who howls at midnight and i will always be crazy and good but i am the bob man i am the mystical food poisoning which one gets on prom night and throws up all night long while the lesbian you brought to prom hates your girlfriend and they presume to bight each others heads off all night long because they are the spawn of the devil whose name begins with baieszselbub i am the spawn whose e name is fish egg mc chicken pants and you shall know us by the trail of dead which spits tobacco out of his face and eats the eternity of my growing head and you see that i am the man whose face is in the shape of a marshmallow and the crackers of his should are in the face of them man who is the mouse pad mc cheese and the man who is in t ehldfsdjljdjd the lavalamp in the brain of the man is interestingly enough the same orgasm of a young boyscout whom saves the squirrel for later if you get my drift talk about safe sex it does not get much safer than having sex with a squirrel squirrels are cool dudes but dudes are not cool squirrels and then you can eat them and they taste quite goodly in a stew pot but don't eat the pot because the pot becomes you and the pot is the pot which is not like other pots but a magical pot of endemic portly proportions which name is nut tickling nipples nancy mcgee and you shall know this pot and know it well you should for in it lies your salvation and undoing for you will fear me for i am the scourge of god if you had not sinned he would not have sent me hither to punish you"
    # r = Readability(text)
    # print(r.flesch().score)
    # print(r.flesch_kincaid().score)
    # print(r.ari().score)
    # print(r.dale_chall().score)
    # print(r.smog().score)
    csv_file = '../MARK1012_Essays_20221028.csv' # Raw Final Data
    df = pd.read_csv(csv_file, encoding='cp1252')
    print(df.columns[0:10])
    

# Starting at the top feature, loop through the top level and if the word appear, set their respective row to exclude true
# def set_feature_exclude():
#     feat_to_sig_feat_df = pd.read_csv('../analysis_data/feat_to_sig_feat.csv')
#     feat_to_sig_feat_df['exclude'] = False
#     # print(feat_to_sig_feat_df)
#     for i, row in feat_to_sig_feat_df.iterrows():
#         to_exclude = row[1].split(', ')
#         for te in to_exclude:
#             # print(te)
#             # print(feat_to_sig_feat_df.loc[feat_to_sig_feat_df['feature'] == te]['exclude'])
#             if feat_to_sig_feat_df.loc[feat_to_sig_feat_df['feature'] == te]['exclude'].bool == True:
#                 # print("SKIP")
#                 continue
#             else:
#                 # print("NO SKIP")
#                 feat_to_sig_feat_df.loc[feat_to_sig_feat_df['feature'] == te].exclude = True

    # print(feat_to_sig_feat_df)

def clean_feat_to_sig_feat():
    pass

if __name__ == "__main__":
    # cronbach_alpha_sig_feat()
    # information_gain()
    # PCA_calc()
    # data_stats()
    sandbox()
    # pairplot_sig_correlation()
    # feature_to_sig_corr_feature()
    # set_feature_exclude()