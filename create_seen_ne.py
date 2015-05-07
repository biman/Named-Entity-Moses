#!/usr/bin/env python
import codecs, pickle, os
import multiprocessing
seen_ne = set()
def get_seen_ne(r_path):
    files=0
    for r_file in os.listdir(r_path):
        rp = codecs.open(os.path.join(r_path,r_file),encoding='utf8',mode='r')
        text = "Start"
        ##Skip First
        while text != "</Sentence>\n":
            text = rp.readline()
        text = "start"
        end = 0
        while text[0] != '</Story>' and end==0 :         ##End of File
            #print "Start new line"  #,words,text,line
            #if not words:
            #    print "Last line of file"
                #time.sleep(2)
            text = rp.readline().strip().split('\t')
            while text[0] != "</Sentence>":
                text = rp.readline().strip().split('\t')
                if len(text) == 4:
                    #NER bracket open- read next word- NER
                    text = rp.readline().strip().split('\t')      #Bracket Open was seen so read next word
                    while len(text) != 1 :
                        if text[1] == "((":
                            text=rp.readline().strip().split('\t')
                            continue
                        seen_ne.add(text[1])
                        text = rp.readline().strip().split('\t')      #Bracket Open was seen so read next word
                else:
                    if text[0] == '</Story>' :
                        #print "end here", r_file
                        #time.sleep(1)
                        end = 1
                        break
                    pass
        files+=1
        if files==50:
            print "50 files done"
            files=0
if __name__=="__main__":
    get_seen_ne('training-hindi')
    pickle.dump(seen_ne, open( "seen_ne.p", "wb" ))
