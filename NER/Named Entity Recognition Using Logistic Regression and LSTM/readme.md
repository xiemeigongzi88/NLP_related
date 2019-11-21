# Named Entity Recognition Using Logistic Regression and LSTM   

  Named Entity Recognition (NER) refers to the task of extracting entities from
text and tagging them with labels. For example, a typical NER system would
find and tag entities in the sentence “Jim bought 300 shares of Acme Corp. in
2006." as:
Jim → Person; 300 → Number; Acme Corp. → Organization; 2006 → Date
For this project, you will use the CoNLL 2003 dataset [1] to perform NER
tagging. This dataset defines 4 tags: Person, Location, Organization and Miscellaneous. Your task is two-fold: given a sentence, extract all relevant named
entities from the task; and assign each named entity with a relevant tag. To perform both tasks together, NER is typically modeled as a BIO tagging problem:
where B indicates the beginning of a new entity span, it indicates the continuation of an entity span and O indicates that the current word or token does not
belong to an entity span. 

Dataset: the CoNLL

## Steps:
  Feature extraction and representation:
  In the dataset, sentences are represented in the CoNLL form, where each new
line indicates the beginning of a new sentence. Within each sentence, tokens
are also new line-separated; each token associated with its gold NER tag. Each
line also contains some additional information like the gold POS tag and the
syntactic head for each token: you may disregard this information and instead
for each token in a sentence, you will extract the following features: the part-ofspeech  tag, the lemma, all hypernyms, hyponyms, holonyms and meronyms; and
any additional features that you can think of (please document these features
in your report)

  Extracting and tagging named entities: 
  Once all relevant features have been extracted (from Task 1), use the Logistic
Regression model to extract all named entities from text and tag them as Person,
Location, Organization or Miscellaneous. Report the accuracy of classification,
precision, recall and F-score for the test set. Also, report the precision, recall
and F-scores for each tag

  NER using deep learning：
  Design a corpus reader and embedding reader. You will use the same
pre-trained word embedding file that you used in the previous project.


## External links:
  https://appliedmachinelearning.blog/2019/04/01/training-deep-learning-based-named-entity-recognition-from-scratch-disease-extraction-hackathon/
  https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede
  https://medium.com/@rohit.sharma_7010/a-complete-tutorial-for-named-entity-recognition-and-extraction-in-natural-language-processing-71322b6fb090
  https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2
  https://dkedar7.github.io/Named%20Entity%20Recognition.html
  https://github.com/Anand-krishnakumar/Information-Extraction-
  https://github.com/huglittlecat88/wiktextract
  https://medium.com/datadriveninvestor/python-data-science-getting-started-tutorial-nltk-2d8842fedfdd
  https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/

## Result:
  The result can be checked in two different files. One is the from “Step 4 ML”, and the other one is from ner_dl.py file.
