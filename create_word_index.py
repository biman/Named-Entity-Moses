#!/usr/bin/env python
import codecs, pickle, os
word_freq_dict= {}
def word_index(i_path):
    hash=100
    #create word frequencies and numeric hashes
    for i_file in os.listdir(i_path):
        #print i_file
        ip=codecs.open(os.path.join(i_path, i_file), encoding='utf8')
        words=ip.readline()
        while(words!=[]):
            words = ip.readline().strip().split()
            for word in words:
                try:
                    word_freq_dict[word][1] += 1
                except KeyError:
                    word_freq_dict[word] = [hash, 1]
                    hash += 1
        ip.close()
if __name__=="__main__":
    word_index('training-hindi-raw')
    pickle.dump(word_freq_dict, open( "word_index_full_training.p", "wb" ))