#!/usr/bin/env python
import codecs, argparse, os
def clean_file(sgm_dir):
    for f_name in os.listdir(sgm_dir):
        with codecs.open(os.path.join(sgm_dir+"_raw",f_name[:-4]),encoding='utf8', mode='w') as op:
            f = codecs.open(os.path.join(sgm_dir,f_name),encoding='utf8')
            for line in f.readlines():
                end_ind = line.find('>')
                le = len(line.strip())
                if line[:4] == "<doc":
                    print le, end_ind
                if le == end_ind + 1:
                    continue
                op.write(line[end_ind+1:-7])
                op.write('\n')
        f.close()
        op.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NER on SVM')
    parser.add_argument('-f', '--sgm_files', default='../data/sgm_test', help='parallel data of wikipedia titles')
    opts = parser.parse_args()
    clean_file(opts.sgm_files)