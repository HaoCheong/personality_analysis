import pandas as pd
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Lower Casing
def lower_casing(essays):
    essays['TEXT'] = essays['TEXT'].str.lower()
    return essays

def remove_stopwords(essays):
    essays['TEXT'] = essays['TEXT'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return essays

def main():
    essays = pd.read_csv('essays.csv',encoding='cp1252')
    
    # Lower Case the given text
    essays = lower_casing(essays)

    # Remove the stopword in the given text
    essays = remove_stopwords(essays)
    
    # Lemmatize = 

    # Print
    print(essays.head())


if __name__ == "__main__":
    main()