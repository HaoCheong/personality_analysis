
# Data cleanup Function (Per Line basis)

# Run on the assumptions that the line has never been processed before,
# Hence clean up is done accordingly per line

import re
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')

def lower_casing(line):
    return line.str.lower()

def remove_stop_words(line):
    return ' '.join([word for word in line.split() if word not in (stop)])

def remove_punctuation(line):
    newline = line.translate(str.maketrans('','', string.punctuation))
    newline = newline.replace(u"\u2019", "")
    return newline

def main():
    pass

if __name__ == "__main__":
    main()
