# Feature Extraction Function (Per Line basis)

# Run on the assumptions that the line has never been processed before,
# Hence clean up is done accordingly per line

import re

# Number of characters in line
def num_of_char(line):
    return len(remove_all_space(line))

# Remove all numbers in line
def remove_num(line):
    return ''.join([i for i in line if not i.isdigit()])

# Remove all the large spaces (" +" -> " ")
def remove_all_large_space(line):
    return re.sub(r' +', ' ', line.strip())

# Remove all white spaces
def remove_all_space(line):
    return re.sub(r' *', '', line.strip())

# Number of long words, defined in regex
def num_long_words(line):
    return len((remove_all_large_space(re.sub(r'\b\w{1,5}\b', '', remove_num(line)))).split(" "))

# Number of short words, defined in regex
def num_short_words(line):
    return len((remove_all_large_space(re.sub(r'\b\w{6,}\b', '', remove_num(line)))).split(" "))

# Number of any words (needs clean up)
def num_any_word(line):
    return len(remove_num(line).split(" "))

# Number of different word
def num_diff_word(line):
    return len(set(line.split(" ")))

# Number of sentences (Tokenised based on ".")
def num_sentences(line):
    return len(line.split('.'))

# Average sentences length (Total word count / sentence count)
def avg_sentence_length(line):
    return float(len(num_any_word(line))/len(num_sentences(line)))

def main():
    pass

if __name__ == "__main__":
    main()
