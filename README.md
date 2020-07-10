
# Naive Bayes and NLP Modeling

Before returning to our Satire/No Satire example, let's consider an example with a smaller but similar scope.

Suppose we are using an API to gather articles from a news website and grabbing phrases from two different types of articles:  music and politics.

We have a problem though! Only some of our articles have their category (music or politics). Is there a way we can use Machine Learning to help us label our data quickly?

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

### How do you think we should calculate: $ P(phrase | politics) $ ?


```python
# we need to break the phrases down into individual words

```

 $\large P(phrase | politics) = \prod_{i=1}^{d} P(word_{i} | politics) $


```python
### We need to make a *Naive* assumption.
```

 $\large P(word_{i} | politics) = \frac{\#\ of\ word_{i}\ in\ politics\ art.} {\#\ of\ total\ words\ in\ politics\ art.} $

### Can you foresee any issues with this?


```python
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

p(phrase|politics)


```python
def vocab_maker(category):
    """returns the vocabulary for a given type of article"""
    vocab_category = set()
    for art in category:
        words = art.split()
        for word in words:
            vocab_category.add(word)
    return vocab_category
        
voc_music = vocab_maker(music)
voc_pol = vocab_maker(politics)
# total_vocabulary = voc_music.union(voc_pol)

```


```python
voc_music
```




    {'a',
     'arena',
     'band',
     'disagreed',
     'for',
     'leaders',
     'on',
     'out',
     'played',
     'popular',
     'sold',
     'song',
     'sound',
     'stadium',
     'the',
     'was'}




```python
voc_music
```




    {'a',
     'arena',
     'band',
     'disagreed',
     'for',
     'leaders',
     'on',
     'out',
     'played',
     'popular',
     'sold',
     'song',
     'sound',
     'stadium',
     'the',
     'was'}




```python
voc_all = voc_music.union(voc_pol)
```


```python
total_vocab_count = len(voc_all)
total_music_count = len(voc_music)
total_politics_count = len(voc_pol)
```


```python
#P(politics | leaders agreed to fund the stadium)
```


```python
def find_number_words_in_category(phrase,category):
    statement = phrase.split()
    str_category=' '.join(category)
    cat_word_list = str_category.split()
    word_count = defaultdict(int)
    for word in statement:
        for art_word in cat_word_list:
            if word == art_word:
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




    defaultdict(int,
                {'world': 0,
                 'leaders': 1,
                 'agreed': 0,
                 'to': 0,
                 'fund': 0,
                 'the': 1,
                 'stadium': 1})




```python
test_politic_word_count = find_number_words_in_category(test_statement,politics)
```


```python
test_politic_word_count
```




    defaultdict(int,
                {'world': 1,
                 'leaders': 1,
                 'agreed': 1,
                 'to': 0,
                 'fund': 0,
                 'the': 2,
                 'stadium': 0})




```python
def find_likelihood(category_count,test_category_count,alpha):
    num = np.product(np.array(list(test_category_count.values())) + alpha)
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

    4.107740405680756e-11
    1.748875897714495e-10


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




    True



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Noting that the resignation of James Mattis as...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Desperate to unwind after months of nonstop wo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nearly halfway through his presidential term, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Attempting to make amends for gross abuses of ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decrying the Senate’s resolution blaming the c...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



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
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
token_docs = [regexp_tokenize(doc, pattern) for doc in X_train]
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
X_t_vec = cv.fit_transform(X_t)
X_t_vec  = pd.DataFrame.sparse.from_spmatrix(X_t_vec)
X_t_vec.columns = sorted(cv.vocabulary_)
X_t_vec.set_index(y_t.index, inplace=True)
```


```python
X_t_vec
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>people</th>
      <th>say</th>
      <th>state</th>
      <th>trump</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>159</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>246</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>640</th>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>809</th>
      <td>2</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>130</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>148</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>356</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>895</th>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>562 rows × 5 columns</p>
</div>




```python
X_val_vec = cv.transform(X_val)
X_val_vec  = pd.DataFrame.sparse.from_spmatrix(X_val_vec)
X_val_vec.columns = sorted(cv.vocabulary_)
X_val_vec.set_index(y_val.index, inplace=True)

```

# Knowledge Check

The word say shows up in our count vectorizer, but it is excluded in the stopwords.  What is going on?

# Multinomial Naive Bayes

Let's break down MNB with our X_t_vec, and y_t arrays in mind.

What are the priors for each class as calculated from these arrays?


```python
np.log(prior_0)
```




    -0.665075161781259



Let's train our model.


```python
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(X_t_vec, y_t)
mnb.__dict__
```




    {'alpha': 1.0,
     'fit_prior': True,
     'class_prior': None,
     'n_features_': 5,
     'classes_': array([0, 1]),
     'class_count_': array([273., 289.]),
     'feature_count_': array([[ 211., 1419.,  371.,  283.,  348.],
            [ 385.,  241.,  111.,  152.,  264.]]),
     'feature_log_prob_': array([[-2.52081091, -0.61898504, -1.95850333, -2.22842295, -2.02232526],
            [-1.09861229, -1.56551193, -2.33595079, -2.02401174, -1.47471983]]),
     'class_log_prior_': array([-0.72203005, -0.66507516])}




```python
# https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
```


```python
random_sample = X_val_vec.sample(1, random_state=40)
random_sample.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>people</th>
      <th>say</th>
      <th>state</th>
      <th>trump</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>520</th>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Our Likelihoods would look like so:

$$ \Large P(satire|count\_people, count\_say...count\_year)$$

$$ \Large P(not\_satire|count\_people, count\_go...count\_year)$$


```python
likelihood_nosat = mnb.feature_log_prob_[0]*random_sample
likelihood_sat =  mnb.feature_log_prob_[1]*random_sample
likelihood_nosat = likelihood_nosat.agg(sum, axis=1)
likelihood_sat = likelihood_sat.agg(sum, axis=1)

print(likelihood_nosat, likelihood_sat)
```

    520   -20.627051
    dtype: float64 520   -27.54762
    dtype: float64



```python
likelihood_nosat + mnb.class_log_prior_[0]
```




    520   -21.349081
    dtype: float64




```python
likelihood_sat + mnb.class_log_prior_[1]
```




    520   -28.212696
    dtype: float64




```python
mnb.predict(random_sample)
```




    array([0])




```python
y_val.loc[random_sample.index]
```




    520    0
    Name: target, dtype: int64




```python

y_hat = mnb.predict(X_val_vec)
y_hat
```




    array([0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
           1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0,
           1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
           1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0,
           0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
           1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
           1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])




```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

accuracy_score(y_val, y_hat)
```




    0.8297872340425532




```python
f1_score(y_val, y_hat)
```




    0.8202247191011236




```python
confusion_matrix(y_val, y_hat)
```




    array([[83, 16],
           [16, 73]])



That performs very well for only having 5 features.

Let's see what happens when we increase our feature set


```python
cv = CountVectorizer()
X_t_vec = cv.fit_transform(X_t)
X_t_vec  = pd.DataFrame.sparse.from_spmatrix(X_t_vec)
X_t_vec.columns = sorted(cv.vocabulary_)
X_t_vec.set_index(y_t.index, inplace=True)
X_t_vec.shape
```




    (562, 14819)




```python
X_val_vec = cv.transform(X_val)
X_val_vec  = pd.DataFrame.sparse.from_spmatrix(X_val_vec)
X_val_vec.columns = sorted(cv.vocabulary_)
X_val_vec.set_index(y_val.index, inplace=True)
```


```python
mnb = MultinomialNB()

mnb.fit(X_t_vec, y_t)
y_hat = mnb.predict(X_val_vec)
y_hat
```




    array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
           0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
           1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
           1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1,
           0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
           0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0])




```python
accuracy_score(y_val, y_hat)
```




    0.9627659574468085




```python
f1_score(y_val, y_hat)
```




    0.96045197740113




```python
confusion_matrix(y_val, y_hat)
```




    array([[96,  3],
           [ 4, 85]])



That performs very well. 

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

f1_score(y_val, y_hat)
```




    0.9378531073446328




```python
X_t_vec.shape
```




    (562, 650)




```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_t_vec, y_t)
rf.score(X_val_vec, y_val)
```




    0.9680851063829787



# Bonus NLP EDA


```python
policies = pd.read_csv('data/2020_policies_feb_24.csv')
policies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>policy</th>
      <th>candidate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>100% Clean Energy for America</td>
      <td>As published on Medium on September 3rd, 2019:...</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A Comprehensive Agenda to Boost America’s Smal...</td>
      <td>Small businesses are the heart of our economy....</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A Fair and Welcoming Immigration System</td>
      <td>As published on Medium on July 11th, 2019:\nIm...</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>A Fair Workweek for America’s Part-Time Workers</td>
      <td>Working families all across the country are ge...</td>
      <td>warren</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A Great Public School Education for Every Student</td>
      <td>I attended public school growing up in Oklahom...</td>
      <td>warren</td>
    </tr>
  </tbody>
</table>
</div>



# Question set 1:
After remove punctuation and ridding the text of numbers and other low semantic value text, answer the following questions.

1. Which document has the greatest average word length?
2. What is the average word length of the entire corpus?
3. Which is greater, the average word length for the documents in the Warren or Sanders campaigns? 



```python

```

Proceed through the remaining standard preprocessing steps in whatever manner you see fit. Make sure to:
- Make text lowercase
- Remove stopwords
- Stem or lemmatize


```python

```

# Question set 2:
1. What are the most common words across the corpus?
2. What are the most common words across each campaign?

> in order to answer these questions, you may find the nltk FreqDist function helpful.

3. Use the FreqDist plot method to make a frequency plot for the corpus as a whole.  
4. Based on that plot, should any more words be added to our stopword library?



```python

```

# Question set 3:

1. What are the most common bigrams in the corpus?
2. What are the most common bigrams in the Warren campain and the Sanders campaign, respectively?
3. Answer questions 1 and 2 for trigrams.

> Hint: You may find it useful to leverage the nltk.collocations functions


```python

```

After answering the questions, transform the data into a document term matrix using either CountVectorizor or TFIDF.  

Run a Multinomial Naive Bayes classifier and judge how accurately our models can separate documents from the two campaigns.


```python

```
