#!/usr/bin/env python
from collections import defaultdict
import codecs, pickle
wiki_freq=defaultdict(int)
def index_wiki_freq(w_file):
    with codecs.open(w_file,encoding='utf8') as w:
        for words in w.readlines():
            words= words.strip().split()
            for word in words:
                if(word=='|||'):
                    break
                wiki_freq[word] +=1
    w.close()
if __name__=="__main__":
    index_wiki_freq('wiki-titles.hi-en')
    pickle.dump(wiki_freq, open( "wiki_freq.p", "wb" ))