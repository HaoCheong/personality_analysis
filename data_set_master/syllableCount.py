import re
import pandas as pd

# Writes the mhyph.txt file into something utf 8 readable
def convert2Readable():
    with open("../data_set_master/mhyph.txt", 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for i in lines:
            newline = re.sub(u'¥','-', i)
            with open('../data_set_master/mhyphFixed.txt', 'a') as g:
                g.write(newline)


# Create a CSV to mhyphFixed of word, to syllable count,
# 2 Types of Delim
def syllableToCSV():
    word_syllable_list = []
    with open("../data_set_master/mhyphFixed.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            new_item = []
            new_item.append(re.sub(r'[-\n]',"", line).lower())
            new_item.append(len(re.split(r'[^\w]',line)) - 1)
            word_syllable_list.append(new_item)

    df = pd.DataFrame(word_syllable_list, columns =['word', 'syllable_count'])
    df2 = df.drop_duplicates(keep='first')
    df2.to_csv('syllable_count.csv', sep=',', encoding='utf-8', index = False)

def main():
    pass
    # syllableToCSV()

if __name__ == "__main__":
    main()