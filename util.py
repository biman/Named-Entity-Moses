#!/usr/bin/env python
from sklearn import svm
import argparse, os, pickle, datetime
import codecs,time
from sklearn.externals import joblib
word_freq_dict= pickle.load(open("word_index_full_training.p","rb"))
suffix_dict = pickle.load(open("suffix_full_training.p","rb"))
prefix_dict = pickle.load(open("prefix_full_training.p","rb"))
word_keys= word_freq_dict.keys()
suffix_keys = suffix_dict.keys()
prefix_keys = prefix_dict.keys()
wiki_vocab = pickle.load(open( "wiki_vocab.p", "rb" ))
train_data_ne = pickle.load(open( "seen_ne.p", "rb" ))
assign = lambda word,feats: feats[-1].append(word_freq_dict[word][0]) if word in word_keys else feats[-1].append(0)
extract_suffix = lambda word: word if len(word)<=3 else word[-3:]
extract_prefix = lambda word: word if len(word)<=3 else word[0:3]
test_words_in_train = []
test_arch_words_in_train = []
test_words = []
def add_feature_vector(words, index, feats, labels, flag):
    feats.append([])
    feats[-1].append(1) if words[index] in train_data_ne else feats[-1].append(0)
    feats[-1].append(1) if words[index] in wiki_vocab else feats[-1].append(0)
    #First word or not
    #try:
    if index == 0:  #float(text[0].split('.')[0])>=1 and float(text[0].split('.')[0])<2):
        feats[-1].append(1)     #['fsword'] = 1
    else:
        feats[-1].append(0)     #['fsword'] = 0
    #except ValueError:
    #    feats[-1].append(0)
    #previous context
    feats[-1].append(0) if index-3 < 0 else assign(words[index-3],feats) #.append(words[index-3])
    feats[-1].append(0) if index-2 < 0 else assign(words[index-2],feats)
    feats[-1].append(0) if index-1 < 0 else assign(words[index-1],feats)
    '''
    if index-2 < 0:
        feats[-1]['prev_context2'] = '0'
    else:
        feats[-1]['prev_context2'] = words[index-2]#.append(words[index-2])
    if index-1 < 0:
        feats[-1]['prev_context1'] = '0'
    else:
        feats[-1]['prev_context1'] = words[index-1]#.append(words[index-1])
    '''
    #next context
    feats[-1].append(0) if index+1 >= len(words) else assign(words[index+1],feats)
    feats[-1].append(0) if index+2 >= len(words) else assign(words[index+2],feats)
    '''
    if index+2 >= len(words):
        feats[-1]['next_context2'] = '0'
    else:
        feats[-1]['next_context2'] = words[index+2]#.append(words[index+2])
    '''
    #print words[index],text[1]
    #suffix
    '''
    if len(words[index]) <= 3:
        feats[-1]['suffix'] = words[index]
    else:
        feats[-1]['suffix'] = words[index][-3:]#.append(words[index][-3:])
    #prefix
    if len(words[index]) <= 3:
        feats[-1]['prefix'] = words[index]
    else:
        feats[-1]['prefix'] = words[index][0:3]#.append(words[index][0:3])
    '''
    w_s=extract_suffix(words[index])
    if w_s in suffix_keys:
        feats[-1].append(suffix_dict[w_s][0])
    else:
        feats[-1].append(0)
    w_p=extract_prefix(words[index])
    if w_p in prefix_keys:
        feats[-1].append(prefix_dict[w_p][0])
    else:
        feats[-1].append(0)
    #NE Information of previous two words
    #This word
    if flag==2:
        feats[-1].append(1) if words[index] in train_data_ne else feats[-1].append(0)
    else:
        feats[-1].append(labels[-1])
    #previous two labels
    if index >= 2:
        if flag==2:             ##Test data- no label available
            feats[-1].append(1) if words[index-2] in train_data_ne else feats[-1].append(0)
            feats[-1].append(1) if words[index-1] in train_data_ne else feats[-1].append(0)
        else:
            feats[-1].append(labels[-3])    #['prev_word2'] = labels[-2]#
            feats[-1].append(labels[-2])    #['prev_word1'] = labels[-1]#
    elif index == 1:
        feats[-1].append(0)    #['prev_word2'] = '0'#.append('0')
        if flag==2:
            feats[-1].append(1) if words[index-1] in train_data_ne else feats[-1].append(0)
        else:
            feats[-1].append(labels[-2])    #['prev_word1'] = labels[-1]#.append(labels[-1])
    else:
        feats[-1].append(0)  #['prev_word2'] = '0'  #.append('0')
        feats[-1].append(0)  #['prev_word1'] = '0'  #.append('0')
    '''POS INFORMATION UNAVAILABLE'''
    #length
    feats[-1].append(len(words[index]))     #['length'] = len(words[index])  #
    #Rare Word
    if words[index] in word_keys:
        if word_freq_dict[words[index]][1] <= 10:    #Limit set for infrequent word
            feats[-1].append(1)     #['rare'] = 1  #.append('1')
        else:
            feats[-1].append(0)     #['rare'] = 0#.append('0')
    else:
        feats[-1].append(1)     #['rare'] = 0#.append('0')
    #Numbers:
    feats[-1].append(1)     #['number'] = 1#.append('1')
    try:
        if '.' in words[index]:
            y='.'
        elif '/' in words[index]:
            y='/'
        else:
            y=' '
        for n in words[index].split(y):
            float(n)
    except ValueError:
        feats[-1][-1]=0     #['number'] = 0#[-1]=0
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
def build_features(i_path, r_path, feats, labels, flag):
    for r_file in os.listdir(r_path):
        rp = codecs.open(os.path.join(r_path,r_file),encoding='utf8',mode='r')
        ip = codecs.open(os.path.join(i_path,r_file+"-1"),encoding='utf8')
        ip.readline()
        text = "Start"
        #print r_file
        ##Skip First
        while text != "</Sentence>\n":
            text = rp.readline()
        text = "start"
        #print "start new file"
        #time.sleep(2)
        end = 0
        line=0
        while text[0] != '</Story>' and end==0 :         ##End of File
            line+=1
            words = ip.readline().strip().split()
            index = -1
            #print "Start new line"  #,words,text,line
            #if not words:
            #    print "Last line of file"
                #time.sleep(2)
            text = rp.readline().strip().split('\t')
            while text[0] != "</Sentence>":
                text = rp.readline().strip().split('\t')
                if len(text) == 4 or len(text) == 2:
                    #print "Just entered",len(text),text,words
                    if len(text) == 4:
                        #NER bracket open- read next word- NER
                        text = rp.readline().strip().split('\t')      #Bracket Open was seen so read next word
                        while len(text) != 1 :
                            if text[1] == "((":
                                text=rp.readline().strip().split('\t')
                                continue
                            index += 1
                            labels.append(1)
                            if flag==1:
                                test_words.append(words[index])
                                if words[index] in train_data_ne:
                                    test_words_in_train.append(len(feats))
                            add_feature_vector(words, index, feats, labels, flag)
                            text = rp.readline().strip().split('\t')      #Bracket Open was seen so read next word
                    elif len(text) == 2:
                        index += 1
                        labels.append(0)
                        #read word- not NER
                        if flag==1:
                            test_words.append(words[index])
                            if words[index] in train_data_ne and len(words[index])>4:
                                test_words_in_train.append(len(feats))       #store index
                        add_feature_vector(words, index, feats, labels, flag)
                else:
                    if text[0] == '</Story>' :
                        print "end here", r_file
                        #time.sleep(1)
                        end = 1
                        break
                    pass
                    #length 1: close bracket or </sen> or <sen id> or \n
                    #length 3: "0 (( SSF"
        #    exit(0)
def build_test_features(t_path,feats,flag):
    for t_file in os.listdir(t_path):
        t = codecs.open(os.path.join(t_path,t_file),encoding='utf8',mode='r')
        text = "Start"
        #print r_file
        text = "start"
        #print "start new file"
        #time.sleep(2)
        end = 0
        for l in t.readlines():        ##End of File
            words = l.strip().split()
            #print "Start new line"  #,words,text,line
            #if not words:
            #    print "Last line of file"
                #time.sleep(2)
            for index in range(len(words)):
                if words[index] in train_data_ne:           #flag always 2 since always called on Test
                    test_arch_words_in_train.append(len(feats))
                test_words.append(words[index])
                add_feature_vector(words, index, feats, [0,0,0], flag)
        #    exit(0)

def train_svm(feats, labels):
    clf = svm.SVC()
    clf.fit(feats, labels)
    joblib.dump(clf, 'models/svm'+datetime.datetime.now().isoformat()+'.pkl')
def test_svm(features, prediction, svm_model):
    prediction.append(svm_model.predict(features))
def compute_metrics(prediction, labels):
    tp = tn = fp = fn = 0
    total = 0
    #print prediction, len(prediction[0]),len(labels)
    for (p, l) in zip(prediction[0], labels):
        total += 1
        #print p , l
        if p < 0.5:
            if l == 0:
                tn += 1
            else:
                fn += 1
        else:
            if l == 1:
                tp += 1
            else:
                fp += 1
    print float(tn+tp)/total
    print "TP=",tp, " TN= ",tn," FP=",fp," FN=",fn