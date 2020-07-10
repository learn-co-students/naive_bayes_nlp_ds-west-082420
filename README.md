
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

### How do you think we should calculate: $ P(phrase | politics) $ ?

 $\large P(phrase | politics) = \prod_{i=1}^{d} P(word_{i} | politics) $

 $\large P(word_{i} | politics) = \frac{\#\ of\ word_{i}\ in\ politics\ art.} {\#\ of\ total\ words\ in\ politics\ art.} $

### Can you foresee any issues with this?

## Laplace Smoothing
 $\large P(word_{i} | politics) = \frac{\#\ of\ word_{i}\ in\ politics\ art. + \alpha} {\#\ of\ total\ words\ in\ politics\ art. + \alpha d} $

 $\large P(word_{i} | music) = \frac{\#\ of\ word_{i}\ in\ music\ art. + \alpha} {\#\ of\ total\ words\ in\ music\ art. + \alpha d} $

This correction process is called Laplace smoothing:
* d : number of features (in this instance total number of vocabulary words)
* $\alpha$ can be any number greater than 0 (it is usually 1)


#### Now let's find this calculation

<img src="./resources/IMG_0041.jpg">

p(phrase|politics)

 $ P(politics | article) = P(politics) x \prod_{i=1}^{d} P(word_{i} | politics) $

#### Deteriming the winner of our model:

<img src = "./resources/solvingforyhat.png" width= "400">

Many times, the probabilities we end up are exceedingly small, so we can transform them using logs to save on computation speed

$\large log(P(politics | article)) = log(P(politics)) + \sum_{i=1}^{d}log( P(word_{i} | politics)) $





Good Resource: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.htmlm

# Back to Satire

Like always, we will perform a train test split...

...and preprocess the training set like we learned.

For demonstration purposes, we will limit our count vectorizer to 5 words (the top 5 words by frequency).

# Knowledge Check

The word say shows up in our count vectorizer, but it is excluded in the stopwords.  What is going on?

# Multinomial Naive Bayes

Let's break down MNB with our X_t_vec, and y_t arrays in mind.

What are the priors for each class as calculated from these arrays?


```python

prior_1 = y_t.value_counts()[0]/len(y_t)
prior_0 = y_t.value_counts()[1]/len(y_t)
print(prior_0, prior_1)
```

    0.5142348754448398 0.48576512455516013


Let's train our model.

Our Likelihoods would look like so:

$$ \Large P(satire|count\_people, count\_say...count\_year)$$

$$ \Large P(not\_satire|count\_people, count\_go...count\_year)$$

That performs very well for only having 5 features.

Let's see what happens when we increase our feature set

That performs very well. 

Let's see whether or not we can maintain that level of accuracy with less words.

# Bonus NLP EDA

# Question set 1:
After remove punctuation and ridding the text of numbers and other low semantic value text, answer the following questions.

1. Which document has the greatest average word length?
2. What is the average word length of the entire corpus?
3. Which is greater, the average word length for the documents in the Warren or Sanders campaigns? 


Proceed through the remaining standard preprocessing steps in whatever manner you see fit. Make sure to:
- Make text lowercase
- Remove stopwords
- Stem or lemmatize

# Question set 2:
1. What are the most common words across the corpus?
2. What are the most common words across each campaign?

> in order to answer these questions, you may find the nltk FreqDist function helpful.

3. Use the FreqDist plot method to make a frequency plot for the corpus as a whole.  
4. Based on that plot, should any more words be added to our stopword library?


# Question set 3:

1. What are the most common bigrams in the corpus?
2. What are the most common bigrams in the Warren campain and the Sanders campaign, respectively?
3. Answer questions 1 and 2 for trigrams.

> Hint: You may find it useful to leverage the nltk.collocations functions

After answering the questions, transform the data into a document term matrix using either CountVectorizor or TFIDF.  

Run a Multinomial Naive Bayes classifier and judge how accurately our models can separate documents from the two campaigns.
