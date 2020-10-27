
# Naive Bayes and NLP Modeling


```python
"""
1.96
"""
```




    '\n1.96\n'



Before returning to our Satire/No Satire example, let's consider an example with a smaller but similar scope.

Suppose we are using an API to gather articles from a news website and grabbing phrases from two different types of articles:  **music** and **politics**.

We have a problem though! Only some of our articles are labeled with a category (music or politics). Is there a way we can use Machine Learning to help us label our data **quickly**?

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

### How do you think we should calculate: $ P(phrase | politics) $ ?


```python
# we need to break the phrases down into individual words

```

 $\large P(phrase | politics) = \prod_{i=1}^{d} P(word_{i} | politics) $

The likelihood of a class label given the phrase is the joint probability distribution of the individual words, or in other words the product of their individual probabilities of appearing in a class.

We need to make a *Naive* assumption.  Naive in this contexts means that we assume that the probabilities of each word appearing are independent from the other words in the phrase.  For example,  the probability of the word 'rock' would increase if we found the word 'classic' in the text.  Naive bayes does not take this conditional probability into account.

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

Let's remind ourselves of the goal, to see the posterior likelihood of the class politics given our phrase. 

> P(politics | leaders agreed to fund the stadium)

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

Now let's fit the the Multinomial Naive Bayes Classifier on our training data


```python

prior_1 = y_t.value_counts()[1]/len(y_t)
prior_0 = y_t.value_counts()[0]/len(y_t)
print(prior_0, prior_1)
print(np.log(prior_1))
```

    0.48576512455516013 0.5142348754448398
    -0.665075161781259


Let's consider the scenario that we would like to isolate satirical news on Facebook so we can flag it.  We do not want to flag real news by mistake. In other words, we want to minimize falls positives.

That's pretty good for a five word vocabulary.

Let's see what happens when we increase don't restrict our vocabulary

Wow! Look how well that performed. 

Let's see whether or not we can maintain that level of accuracy with less words.

TFIDF does not necessarily perform better than CV.  It is just a tool in our toolbelt which we can try out and compare the performance.  

Let's compare MNB to one of our classifiers that has a track record of high performance, Random Forest.

Both random forest and mnb perform comparably, however, mnb is lightweight as far as computational power and speed.  For real time predictions, we may choose MNB over random forest because the classifications can be performed quickly.

