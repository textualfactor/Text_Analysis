import os
import time
from bs4 import BeautifulSoup
import re
import codecs
import pandas as pd


#regex = re.compile('[%s]' % re.escape(string.punctuation.replace('-','')+control_chars))

def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)
    
def clean_punc(s):
    return re.sub("[^A-Za-z\-]", "", s).lower()


def tokenize(wdir, output):
    start = time.time()
    for num, file in enumerate(sorted(os.listdir(wdir))):
        try:
            openfile = codecs.open(wdir + file, 'r', encoding='utf-8', errors='ignore')
            file_name= '99_' + file[:10]
            text = openfile.read()
            soup = BeautifulSoup(text,'html.parser')
            raw = BeautifulSoup.get_text(soup)
            words = raw.split()
            word_dic = {}
            for word in words:
                if (not hasNumbers(word)) and word[0]!='-' and word[-1]!='-':
                    word = clean_punc(word)
                    if word not in word_dic:
                        word_dic[word] = 1
                    else:
                        word_dic[word] += 1
            openfile.close()
            outfile = open(output, 'a')
            for key, value in word_dic.items():
                outfile.write(file_name+','+key+','+str(value)+'\n')
            outfile.close()
            if num%100 == 0:
                print(str(num)+' done!  time:'+str(time.time()-start))
        except:
            print('file {} skipped'.format(num))
    df = pd.read_csv('output_mdna99.csv')
    df = df.dropna()
    df.to_csv('output_mdna99.csv',index=False) 
    
    


        
