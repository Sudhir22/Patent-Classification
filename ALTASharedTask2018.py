# #**ALTA Shared Task 2018**


import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neural_network
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


bow_vectorizer = CountVectorizer(lowercase = False, 
                                     tokenizer = lambda x: x, # because we already have tokens available
                                     stop_words = None, ## stop words removal already done from NLTK
                                     max_features = 150000, ## pick top 100K words by frequency
                                     ngram_range = (1, 2), ## we want bigrams now
                                     binary = False) ## we do not want as binary/boolean features

# ##**Converting description into tokens (Tokenizing Data)**
def preprocessor(text):
    __tokenization_pattern = r'''(?x)          # set flag to allow verbose regexps
        \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''

    ## call it using tokenizer.tokenize
    tokenizer = nltk.tokenize.regexp.RegexpTokenizer(__tokenization_pattern)
    tokens = tokenizer.tokenize(text.lower())
    alphabet_tokens = [token for token in tokens if token.isalpha()]
    en_stopwords = set(nltk.corpus.stopwords.words('english'))
    non_stopwords = [word for word in alphabet_tokens if not word in en_stopwords]
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    stems = [str(stemmer.stem(word)) for word in non_stopwords]
    
    return stems


# ##**Preparing dataframe for the training data**

def prepare_data(path):
    data=pd.read_csv(path)
    description=list()
    for id in data.id:
        with open('patents/'+str(id)+'.txt',encoding='utf8',errors='ignore') as f:
            x=f.read()
            description.append(x.strip())
    data['description']=description
    data.first_ipc_mark_section=[ord(x)-64 for x in data.first_ipc_mark_section]
    data['description_tokens'] = data['description'].apply(preprocessor)
    return data



# ##**Create an ML Pipeline**

def create_pipeline(data):
    c=Pipeline(steps=[('bow',bow_vectorizer),
                    ('tfidf',TfidfTransformer()),
                    ('lr',LogisticRegression(C=40))])


    msk = np.random.rand(len(data)) < 0.75
    train_X = data.description_tokens[msk]
    test_X = data.description_tokens[~msk]
    y= data['first_ipc_mark_section']
    train_y = y[msk]
    test_y = y[~msk]

    c.fit(train_X,train_y)
    return c


# ##**Using trained model for prediction/inference**

if __name__=='__main__':
    train_data = prepare_data('train.csv')
    test_data = prepare_data('test.csv')
    model = create_pipeline(train_data)
    test_data['first_ipc_mark_section'] = model.predict(test_data.description_tokens)
    test_data.first_ipc_mark_section=[chr(x+64) for x in test_data.first_ipc_mark_section]
    test_data[['id','first_ipc_mark_section']].to_csv('output.csv',index=None)
















