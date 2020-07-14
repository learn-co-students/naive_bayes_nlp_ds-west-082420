
# Naive Bayes and NLP Modeling

Before returning to our Satire/No Satire example, let's consider an example with a smaller but similar scope.

Suppose we are using an API to gather articles from a news website and grabbing phrases from two different types of articles:  **music** and **politics**.

We have a problem though! Only some of our articles have their category (music or politics). Is there a way we can use Machine Learning to help us label our data **quickly**?

-------------------------------
### Here are our articles
#### Music Articles:

* 'the song was popular'
* 'band leaders disagreed on sound'
* 'played for a sold out arena stadium'

#### Politics Articles

* 'world leaders met lask week'
* 'the election was close'
* 'the officials agreed on a compromise'
--------------------------------------------------------
Let's try and predict one example phrase:


* "world leaders agreed to fund the stadium"

How can we make a model that labels this for us rather than having to go through by hand?


```python
from collections import defaultdict
import numpy as np
music = ['the song was popular',
         'band leaders disagreed on sound',
         'played for a sold out arena stadium']

politics = ['world leaders met lask week',
            'the election was close',
            'the officials agreed on a compromise']

test_statement = 'world leaders agreed to fund the stadium'
```


```python
#labels : 'music' 'politics'
#features: words
test_statement_2 = 'officials met at the arena'
```

Let's revisit Bayes Theorem.  Remember, Bayes looks to calculate the probability of a class (c) given the data (x).  To do so, we calculate the **likelihood** (the distribution of our data within a given class) and the **prior** probabiliity of each class (the probability of seeing the class in the population). We are going to ignore the denominator of the right side of the equation in this instance, because, as we will see, we will be finding the ratio of posteriors probabilities, which will cancel out the denominator.

<img src ="./resources/naive_bayes_icon.png">

### Another way of looking at it
<img src = "./resources/another_one.png">

## So, in the context of our problem......



$\large P(politics | phrase) = \frac{P(phrase|politics)P(politics)}{P(phrase)}$

$\large P(politics) = \frac{ \# politics}{\# all\ articles} $

*where phrase is our test statement*

<img src = "./resources/solving_theta.png" width="400">

### How should we calculate P(politics)?

This is essentially the distribution of the probability of either type of article. We have three of each type of article, therefore, we assume that there is an equal probability of either article


```python
p_politics = len(politics)/(len(politics) + len(music))
p_music = len(music)/(len(politics) + len(music))
```


```python
# This is always a good idea
%load_ext autoreload
%autoreload 2

import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

mccalister = ['Adam', 'Amanda','Chum', 'Dann',
 'Jacob', 'Jason', 'Johnhoy', 'Karim',
'Leana','Luluva', 'Matt', 'Maximilian','Syd' ]

from src.student_caller import one_random_student
```

### How do you think we should calculate: $ P(phrase | politics) $ ?


```python
one_random_student(mccalister)
```


```python
#__SOLUTION__
# we need to break the phrases down into individual words

```

 $\large P(phrase | politics) = \prod_{i=1}^{d} P(word_{i} | politics) $


```python
### We need to make a *Naive* assumption.
```

 $\large P(word_{i} | politics) = \frac{\#\ of\ word_{i}\ in\ politics\ art.} {\#\ of\ total\ words\ in\ politics\ art.} $

### Can you foresee any issues with this?


```python
one_random_student(mccalister)
```


```python
#__SOLUTION__
# we can't have a probability of 0

```

## Laplace Smoothing
 $\large P(word_{i} | politics) = \frac{\#\ of\ word_{i}\ in\ politics\ art. + \alpha} {\#\ of\ total\ words\ in\ politics\ art. + \alpha d} $

 $\large P(word_{i} | music) = \frac{\#\ of\ word_{i}\ in\ music\ art. + \alpha} {\#\ of\ total\ words\ in\ music\ art. + \alpha d} $

This correction process is called Laplace smoothing:
* d : number of features (in this instance total number of vocabulary words)
* $\alpha$ can be any number greater than 0 (it is usually 1)


#### Now let's find this calculation

<img src="./resources/IMG_0041.jpg">


```python
def vocab_maker(category):
    """
    parameters: category is a list containing all the articles
    of a given category.
    
    returns the vocabulary for a given type of article
    
    """
    
    vocab_category = set() # will filter down to only unique words
    
    for art in category:
        words = art.split()
        for word in words:
            vocab_category.add(word)
    return vocab_category
        
voc_music = vocab_maker(music)
voc_pol = vocab_maker(politics)

```


```python
# These are all the unique words in the music category
voc_music
```


```python
# These are all the unique words in the politics category
voc_pol
```


```python
# The union of the two sets gives us the unique words across both article groups
voc_all = voc_music.union(voc_pol)
voc_all
```


```python
total_vocab_count = len(voc_all)
total_music_count = len(voc_music)
total_politics_count = len(voc_pol)
```

Let's remind ourselves of the goal, to see the posterior likelihood of the class politics given our phrase. 

> P(politics | leaders agreed to fund the stadium)


```python
music
```


```python
def find_number_words_in_category(phrase,category):
    statement = phrase.split()
    
    # category is a list of the raw documents of each category
    str_category=' '.join(category)
    cat_word_list = str_category.split()
    word_count = defaultdict(int)
    
    # loop through each word in the phrase
    for word in statement:
        # loop through each word in the category
        for art_word in cat_word_list:
            if word == art_word:
                # count the number of times the phrase word occurs in the category
                word_count[word] +=1
            else:
                word_count[word]
    return word_count
                
            
```


```python
test_music_word_count = find_number_words_in_category(test_statement,music)

```


```python
test_music_word_count
```


```python
test_politic_word_count = find_number_words_in_category(test_statement,politics)
```


```python
test_politic_word_count
```


```python
def find_likelihood(category_count,test_category_count,alpha):
    # The numerator will be the product of all the counts 
    # with the smoothing factor (alpha) to make sure the probability is not zero'd out
    num = np.product(np.array(list(test_category_count.values())) + alpha)
    
    # The denominator will be the same for each word (total category count + total vocab + alph)
    # so we raise it to the power of the length of the test category
    denom = (category_count + total_vocab_count*alpha)**(len(test_category_count))
    
    return num/denom
```


```python
likelihood_m = find_likelihood(total_music_count,test_music_word_count,1)
```


```python
likelihood_p = find_likelihood(total_politics_count,test_politic_word_count,1)
```


```python
print(likelihood_m)
print(likelihood_p)
```

 $ P(politics | article) = P(politics) x \prod_{i=1}^{d} P(word_{i} | politics) $

#### Deteriming the winner of our model:

<img src = "./resources/solvingforyhat.png" width= "400">


```python
p_politics = .5
p_music = .5
```


```python
# p(politics|article)  > p(music|article)
likelihood_p * p_politics  > likelihood_m * p_music
```

Many times, the probabilities we end up are exceedingly small, so we can transform them using logs to save on computation speed

$\large log(P(politics | article)) = log(P(politics)) + \sum_{i=1}^{d}log( P(word_{i} | politics)) $





Good Resource: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.htmlm

# Back to Satire


```python
import pandas as pd
import numpy as np
corpus = pd.read_csv('data/satire_nosatire.csv')
corpus.head()
```

Like always, we will perform a train test split...


```python
X=corpus.body
y=corpus.target
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=.25)
```

...and preprocess the training set like we learned.


```python
import nltk
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
```


```python
# Import our regex pattern that gets rid of numbers and punctuation

sw = stopwords.words('english')
sw.extend(['would', 'one', 'say'])

```


```python
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer 
  

def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
```


```python
def doc_preparer(doc, stop_words=sw):
    '''
    
    :param doc: a document from the satire corpus 
    :return: a document string with words which have been 
            lemmatized, 
            parsed for stopwords, 
            made lowercase,
            and stripped of punctuation and numbers.
    '''
    
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in stop_words]
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    lemmatizer = WordNetLemmatizer() 
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    return ' '.join(doc)
```


```python
token_docs = [doc_preparer(doc, sw) for doc in X_train]
```


```python
from sklearn.feature_extraction.text import CountVectorizer
```

For demonstration purposes, we will limit our count vectorizer to 5 words (the top 5 words by frequency).


```python
# Secondary train-test split to build our best model
X_t, X_val, y_t, y_val = train_test_split(token_docs, y_train, test_size=.25, random_state=42)
```


```python
cv = CountVectorizer(max_features=5)

# Just like with our scaler, we fit our Count Vectorizer on the training set
X_t_vec = cv.fit_transform(X_t)
X_t_vec  = pd.DataFrame.sparse.from_spmatrix(X_t_vec)
X_t_vec.columns = sorted(cv.vocabulary_)
X_t_vec.set_index(y_t.index, inplace=True)
```


```python
X_t_vec
```

# Knowledge Check

The word say shows up in our count vectorizer, but it is excluded in the stopwords.  What is going on?


```python
# We then transform the validation set.  Do not refit the vectorizer
X_val_vec = cv.transform(X_val)
X_val_vec  = pd.DataFrame.sparse.from_spmatrix(X_val_vec)
X_val_vec.columns = sorted(cv.vocabulary_)
X_val_vec.set_index(y_val.index, inplace=True)

```

# Multinomial Naive Bayes

Now let's fit the the Multinomial Naive Bayes Classifier on our training data


```python
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(X_t_vec, y_t)
```


```python
#What should our priors for each class be?

one_random_student(mccalister)
```


```python
#__SOLUTION__

prior_1 = y_t.value_counts()[1]/len(y_t)
prior_0 = y_t.value_counts()[0]/len(y_t)
print(prior_0, prior_1)
print(np.log(prior_1))
```


```python
mnb.class_log_prior_
```


```python
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

y_hat = mnb.predict(X_val_vec)
accuracy_score(y_val, y_hat)
```

Let's consider the scenario that we would like to isolate satirical news on Facebook so we can flag it.  We do not want to flag real news by mistake. In other words, we want to minimize falls positives.


```python
confusion_matrix(y_val, y_hat)
```


```python
precision_score(y_val, y_hat)
```

That's pretty good for a five word vocabulary.

Let's see what happens when we increase don't restrict our vocabulary


```python
cv = CountVectorizer()
X_t_vec = cv.fit_transform(X_t)
X_t_vec  = pd.DataFrame.sparse.from_spmatrix(X_t_vec)
X_t_vec.columns = sorted(cv.vocabulary_)
X_t_vec.set_index(y_t.index, inplace=True)


X_val_vec = cv.transform(X_val)
X_val_vec  = pd.DataFrame.sparse.from_spmatrix(X_val_vec)
X_val_vec.columns = sorted(cv.vocabulary_)
X_val_vec.set_index(y_val.index, inplace=True)
```


```python
mnb = MultinomialNB()

mnb.fit(X_t_vec, y_t)
y_hat = mnb.predict(X_val_vec)
confusion_matrix(y_val, y_hat)
```

Wow! Look how well that performed. 


```python
precision_score(y_val, y_hat)
```


```python
len(cv.vocabulary_)
```

Let's see whether or not we can maintain that level of accuracy with less words.


```python
cv = CountVectorizer(min_df=.05, max_df=.95)
X_t_vec = cv.fit_transform(X_t)
X_t_vec  = pd.DataFrame.sparse.from_spmatrix(X_t_vec)
X_t_vec.columns = sorted(cv.vocabulary_)
X_t_vec.set_index(y_t.index, inplace=True)

X_val_vec = cv.transform(X_val)
X_val_vec  = pd.DataFrame.sparse.from_spmatrix(X_val_vec)
X_val_vec.columns = sorted(cv.vocabulary_)
X_val_vec.set_index(y_val.index, inplace=True)

mnb = MultinomialNB()

mnb.fit(X_t_vec, y_t)
y_hat = mnb.predict(X_val_vec)

precision_score(y_val, y_hat)
```


```python
len(cv.vocabulary_)
```


```python
# Now let's see what happens with TF-IDF
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_t_vec = tfidf.fit_transform(X_t)
X_t_vec  = pd.DataFrame.sparse.from_spmatrix(X_t_vec)
X_t_vec.columns = sorted(tfidf.vocabulary_)
X_t_vec.set_index(y_t.index, inplace=True)

X_val_vec = tfidf.transform(X_val)
X_val_vec  = pd.DataFrame.sparse.from_spmatrix(X_val_vec)
X_val_vec.columns = sorted(tfidf.vocabulary_)
X_val_vec.set_index(y_val.index, inplace=True)

mnb = MultinomialNB()

mnb.fit(X_t_vec, y_t)
y_hat = mnb.predict(X_val_vec)

precision_score(y_val, y_hat)
```

TFIDF does not necessarily perform better than CV.  It is just a tool in our toolbelt which we can try out and compare the performance.  


```python
len(tfidf.vocabulary_)
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=.05, max_df=.95)
X_t_vec = tfidf.fit_transform(X_t)
X_t_vec  = pd.DataFrame.sparse.from_spmatrix(X_t_vec)
X_t_vec.columns = sorted(tfidf.vocabulary_)
X_t_vec.set_index(y_t.index, inplace=True)

X_val_vec = tfidf.transform(X_val)
X_val_vec  = pd.DataFrame.sparse.from_spmatrix(X_val_vec)
X_val_vec.columns = sorted(tfidf.vocabulary_)
X_val_vec.set_index(y_val.index, inplace=True)

mnb = MultinomialNB()

mnb.fit(X_t_vec, y_t)
y_hat = mnb.predict(X_val_vec)

precision_score(y_val, y_hat)
```


```python
len(tfidf.vocabulary_)
```

Let's compare MNB to one of our classifiers that has a track record of high performance, Random Forest.


```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, max_features=5, max_depth=5)
rf.fit(X_t_vec, y_t)
y_hat = rf.predict(X_val_vec)
precision_score(y_val, y_hat)
```

Both random forest and mnb perform comparably, however, mnb is lightweight as far as computational power and speed.  For real time predictions, we may choose MNB over random forest because the classifications can be performed quickly.

