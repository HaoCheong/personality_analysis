
# Data cleanup Function (Per Line basis)

# Run on the assumptions that the line has never been processed before,
# Hence clean up is done accordingly per line

import re
import string
from nltk.corpus import stopwords

stop = stopwords.words('english')

# Lower case the line
def lower_casing(line):
    return line.str.lower()

# Remove all the stop words
def remove_stop_words(line):
    return ' '.join([word for word in line.split() if word not in (stop)])

# Remove punctuation in the line
def remove_punctuation(line):
    newline = line.translate(str.maketrans('','', string.punctuation))
    newline = newline.replace(u"\u2019", "")
    return newline

# Remove all numbers in line
def remove_num(line):
    return ''.join([i for i in line if not i.isdigit()])

# Remove all the large spaces (" +" -> " ")
def remove_all_large_space(line):
    return re.sub(r' +', ' ', line.strip())

# Remove all white spaces
def remove_all_space(line):
    return re.sub(r' *', '', line.strip())

def main():
    pass

if __name__ == "__main__":
    main()
