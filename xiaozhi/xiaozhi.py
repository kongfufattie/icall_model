import pandas as pd
import nltk
import jieba
df=pd.read_excel('keywords.xlsx')
for word in df.word:
    jieba.add_word(word)
tags=[]
with open ('corpus.csv') as f:
    for line in f:
        tags+=list(jieba.cut(line))
dist=nltk.FreqDist(tags)
with open ('freqDist.csv','w') as f:
    f.write('word,count\n')
    for i in dist:
        f.write('{},{}\n'.format(i,dist[i]))