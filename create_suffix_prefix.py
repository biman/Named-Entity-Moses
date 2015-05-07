#!/usr/bin/env python
import codecs, pickle, os
suffix_dict = {}
prefix_dict = {}
extract_suffix = lambda word: word if len(word)<=3 else word[-3:]
extract_prefix = lambda word: word if len(word)<=3 else word[0:3]
def suffix_prefix_index(i_path):
    hash_p = 100.2
    hash_s = 100.8
    #create word frequencies and numeric hashes
    for i_file in os.listdir(i_path):
        #print i_file
        ip=codecs.open(os.path.join(i_path, i_file), encoding='utf8')
        words=ip.readline()
        while(words!=[]):
            words = ip.readline().strip().split()
            for word in words:
                w_s = extract_suffix(word)
                w_p = extract_prefix(word)
                try:
                    suffix_dict[w_s][1] += 1
                except KeyError:
                    suffix_dict[w_s] = [hash_s, 1]
                    hash_s += 1
                try:
                    prefix_dict[w_p][1] += 1
                except KeyError:
                    prefix_dict[w_p] = [hash_p, 1]
                    hash_p += 1
        ip.close()
if __name__=="__main__":
    suffix_prefix_index('training-hindi-raw')
    pickle.dump(suffix_dict, open( "suffix_full_training.p", "wb" ))
    pickle.dump(prefix_dict, open( "prefix_full_training.p", "wb" ))