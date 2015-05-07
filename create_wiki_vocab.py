#!/usr/bin/env python
import codecs, pickle
wiki_vocab=set()
def index_wiki_vocab(w_file):
    with codecs.open(w_file,encoding='utf8') as w:
        for words in w.readlines():
            words= words.strip().split()
            for word in words:
                if(word=='|||'):
                    break
                wiki_vocab.add(word)
    w.close()
if __name__=="__main__":
    index_wiki_vocab('wiki-titles.hi-en')
    pickle.dump(wiki_vocab, open( "wiki_vocab.p", "wb" ))