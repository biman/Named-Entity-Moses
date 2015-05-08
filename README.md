# Named-Entity-Moses
Build Named Entity Recognition for Moses

/src:

util.py: Carries methods for feature bulding, training and testing using scikit-learn. It makes use of pickled files from the /pickled directory.

train.py: Uses training data from /data directory by default. It uses methods for feature building and train from util. It outputs a trained model that is named uniquely using timestamp and is stored in /models directory.

test.py: Uses test and reference data from /data directory by default. It uses methods for feature building and scikit-learn predict from util. It outputs the number of correctly and incorrectly identified named entities as well as a file in the /data directory by the name "ne_words.hi".

test_no_label.py: Same as test.py. Used when there is no reference data. It does not output any evaluation mesures. It only outputs the "ne_words.hi" file.


/scripts: 

clean_sgm_file.py: Used to remove xml tags from the SGML test file obtained from WMT 2014. Raw version of this file is used to create named entitity list to be used with Moses's transliteration model.

create_seen_ne.py: Creates a pickle file comprising a list of all words annotated as named entities in the training data.

create_suffix_prefix.py: Creates a pickle file comprising numerical hash values for all suffixes and prefixes sees in training data so that they can be used as features.

create_wiki_freq.py: Creates a pickle file comprising dictionary object of all words seen in wikipedia titles and their frequency.

create_wiki_vocab.py: Same as create_wiki_freq.py except that it comprises a list object without the frequency information.

create_word_index.py: Same as create_suffix_prefix.py except that it creates hashes and also collects frequencies for full words seen in training data.


/pickled: 

Pickle files created by their corresponding script. These are loaded in util.py for feature building. 


/data:

training_hindi: 176 files comprising Hindi data with named entity annotations obtained from IJCNLP 2007.

training_hindi_raw: Has raw forms of the files corresponding to those in /training_hindi.

test: Reference data for computing evaluation measure on test set.

test-raw: Raw form of test data used for making predictions.

test_no_lavel: Hindi test data obtained from removing XML Tags in sgm_test using clean_sgm_file.py.

wiki-titles.hi-en: File containing parallel Hindi-English Wikipedia titles.

sgm_test: SGML format data obtained from WMT 2014 test set for Hindi-English.

sgm_test_raw: Files obtained after removing XML tags from SGML format files using clean_sgm_file.py.


/models:

Stores model files created by train.py.