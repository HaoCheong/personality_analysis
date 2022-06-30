
# Data cleanup Function (Per Line basis)

# Run on the assumptions that the line has never been processed before,
# Hence clean up is done accordingly per line

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words

stop = stopwords.words('english')
correct_words = words.words()

# Lower case the line
def lower_casing(line):
    return line.lower()

# Remove all the stop words
def remove_stop_words(line):
    return ' '.join([word for word in line.split() if word not in (stop)])

def replace_to_counterpart(line):

    # Replace unicode substitute with utf8 equiv
    newline = line.replace(u"\u2019", "'")
    newline = newline.replace(u"\u2018", "'")
    newline = newline.replace(u"\u8217", "'")

    # Replace - with " "
    newline = newline.replace("-"," ")
    return newline

# Remove all weird characters:
# [\\\\[\]@a-zA-Z0-9.,\/#!$%\^&\*;:"\'{}<>=+\-_`~()\? ]*, all "normal" characters regex, could come in handy later

def remove_weird_char(line):
    newline = re.sub(r'[^a-zA-Z0-9.,\/#!$%\^&\*;:"\'{}=\-_`~()\? ]*', '', line)
    return newline

#Spelling fix, jaccard distrance method, results dubious (NOT USED)
def spelling_fix(line):
    newline = remove_all_large_space(line)
    line_token = newline.split(" ")
    # print(line_token)
    corrected = []
    for word in line_token:
        if (len(word) > 3):
            temp = [(jaccard_distance(set(ngrams(word, 4)),set(ngrams(w, 4))),w) for w in correct_words if w[0]==word[0]]
            corrected.append(sorted(temp, key = lambda val:val[0])[0][1])
        else:
            corrected.append(word)
        
    return " ".join(corrected)



# Remove punctuation in the line, sentence_punc true includes sentence ender punctuation
def remove_punctuation(line, sentence_end = True):
    
    newline = ""

    if (sentence_end):
        newline = re.sub(r'[.,\/#!$%\^&\*;:"\'{}=_`~()\?]*',"",line) # Include sentence enders
    else:
        newline = re.sub(r'[,\/#$%\^&\*;:"\'{}=_`~()]*',"",line) # Exclude sentence enders

    return newline

# Remove all numbers in line
def remove_num(line):
    return ''.join([i for i in line if not i.isdigit()])

# Remove all the large spaces (" +" -> " ")
def remove_all_large_space(line):
    return re.sub(r' +', ' ', line.strip())

# Remove all white spaces
def remove_all_space(line):
    return re.sub(r' ', '', line.strip())
