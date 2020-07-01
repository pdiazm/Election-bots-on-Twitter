# -*- coding: utf-8 -*-
"""
LDA analysis

Created on Sat May 30 11:57:28 2020

@author: Paulino Diaz
"""

#%% Load libraries
import lucem_illud_2020 as lucem #pip install -U git+git://github.com/Computational-Content-Analysis-2020/lucem_illud_2020.git

import time
#All these packages need to be installed from pip
#These are all for feature generation
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
from sklearn.externals import joblib 

import scipy #For comparing distributions and measuring divergence
import gensim#For topic modeling
import numpy as np #for arrays
import pandas as pd #gives us DataFrames
import matplotlib.pyplot as plt #For graphics

from operator import itemgetter #for extracting the topic loadings

from gensim.models.coherencemodel import CoherenceModel #for topic modelling



#%% Load data
#load tweets for all days
days = pd.read_csv(r'.\Classified\all.csv')
days = days.reset_index().drop(['index','Unnamed: 0'], axis=1)

#%% subset variables
tweets = days[['screen_name','status_id','created_at', 'text','cap', 'is_retweet',
               'favorite_count', 'retweet_count', 'followers_count', 'friends_count']]

# keep only original text
tweets = tweets.drop_duplicates(subset='text')

#%% tokenize and normalize the text of each tweet
tweets['tokenized_text'] = tweets['text'].apply(lambda x: lucem.word_tokenize(x))
tweets['normalized_tokens'] = tweets['tokenized_text'].apply(lambda x: lucem.normalizeTokens(x, 
                                                                                             extra_stop=['amp']))

#%% create a subset of max 1000 words that appear less than 3 times and in more...
#...than half of documents
#initialize model
twTFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5,
                                                                 max_features=1500, 
                                                                 min_df=100, 
                                                                 stop_words='english', 
                                                                 norm='l2')
#train the model
twTFVects = twTFVectorizer.fit_transform(tweets['text'])
print(twTFVects.shape)

#%% creat dropMissing function to apply the tf-idf filter
def dropMissing(wordLst, vocab):
    return [w for w in wordLst if w in vocab]

#%% create column with weighted tokens only
tweets['reduced_tokens'] = tweets['normalized_tokens'].apply(lambda x: dropMissing(x,
                                                                                   twTFVectorizer.vocabulary_.keys()))

#%% transform the tokens into a corpus
dictionary = gensim.corpora.Dictionary(tweets['reduced_tokens'])
corpus = [dictionary.doc2bow(text) for text in tweets['reduced_tokens']]
gensim.corpora.MmCorpus.serialize('twitter.mm', corpus)
twittermm = gensim.corpora.MmCorpus('twitter.mm')

#%% define a function to train different models and compute their coherence scores
def score_models(corpus,dictionary,texts,l_of_topics,alpha=.001,eta=.001):
    coherence_values = []
    model_list = []
    for topics in l_of_topics:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=topics,
                                                random_state = 16,
                                                alpha=alpha, 
                                                eta=eta)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, 
                                        texts=texts, 
                                        dictionary=dictionary, 
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
    return model_list, coherence_values

#%% train different models for different # of topics using the above function
# choose a low alpha and beta to get topics with a few distinctive words,
# and documents with only a few distinctive topics
n_topics = list(range(5,21))
models, coherence = score_models(corpus=twittermm,
                                 dictionary=dictionary,
                                 texts = tweets['reduced_tokens'],
                                 l_of_topics = n_topics)

#%% plot the coherence for each topic size
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(n_topics,coherence,
        color='tab:cyan',
        linewidth=4)
ax.axvline(14,ymin=0,ymax=.95,linestyle='dashed',
          linewidth=2, label='max value')
ax.set_title('Figure 2: Coherence Scores for Different # of Topics',
             fontsize=20)
ax.set_ylabel('Coherence',
              fontsize=18)
ax.set_xlabel('# of Topics',
              fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("num_topics_6.5.png", format = 'png', bbox_inches='tight')

#%% save the model
joblib.dump(models[9], r'.\LDA\lda_14.pkl') 

#%% look at the results of the best model
lda_14 = joblib.load(r'.\LDA\lda_14.pkl')  
topicsDict = {}
for topicNum in range(lda_14.num_topics):
    topicWords = [w for w, p in lda_14.show_topic(topicNum)]
    topicsDict['Topic_{}'.format(topicNum)] = topicWords
wordRanks_14 = pd.DataFrame(topicsDict)
wordRanks_14

#%% create a new dataframe with the topic information
#save the topics to a dataframe
ldaDF_14 = pd.DataFrame({
        'created_at' : tweets['created_at'], 
        'status_id' : tweets['status_id'],
        'text' : tweets['text'],
        'topics' : [lda_14[dictionary.doc2bow(l)] for l in tweets['reduced_tokens']]
    })


#%% label tweets based on the highest loading and save the dataframe
ldaDF_14['max_topic'] = ldaDF_14['topics'].apply(lambda x: max(x, key = itemgetter(1))[0])

topics_df = ldaDF_14[['text','max_topic']]
topics_df.to_csv(r'.\LDA\topic_labels.csv')
#%% expand topics into matrix

#Dict to temporally hold the probabilities
topicsProbDict = {i : [0] * len(ldaDF_14) for i in range(lda_14.num_topics)}

#Load them into the dict
for index, topicTuples in enumerate(ldaDF_14['topics']):
    for topicNum, prob in topicTuples:
        topicsProbDict[topicNum][index] = prob
      

#Update the DataFrame
for topicNum in range(lda_14.num_topics):
    ldaDF_14['topic_{}'.format(topicNum)] = topicsProbDict[topicNum]


#%% estimate influence of tweets using KL divergence

# order tweets from earliest to latest
ldaDF_14 = ldaDF_14.reset_index().drop('index', axis=1).sort_values('created_at',
                                                                    ascending=True)

# Transform loadings that are equal to 0 to avoid errors when computing KL divergence
for column in range(4,18):
    ldaDF_14.iloc[:,column] = ldaDF_14.iloc[:,column] + .000000001

# create columns for the values of novelty and transience
ldaDF_14['novelty'] = np.nan
ldaDF_14['transience'] = np.nan

#save the df for easy reloading
#ldaDF_14.to_csv(r'.\Scores\topic_probs.csv')
#ldaDF_14 = pd.read_csv(r'.\Scores\topic_probs.csv').drop('Unnamed: 0', axis=1)

#get the average tweets per minute
#tweets_per_day = pd.DatetimeIndex(ldaDF_14.created_at).day
tweets_per_day = 7394 #pd.Series(tweets_per_day).value_counts().mean()
tweets_per_hour = 308 #tweets_per_day / 24
tweets_per_minute = 5 #tweets_per_hour / 60

#choose a window
window = 1000 #1000 tweets approximate a 3 hour windows

#compute the scores looping over each tweet
for i in range(0+window, len(ldaDF_14)-window):
    novelties=[]
    transiences = []
    for neighbor in range(1, window+1):
        start = time.time() #time the loop
        # save the topic loadings into an array of probabilities
        probs1 = pd.array(ldaDF_14.iloc[i,4:18], dtype='float') #for target doc
        probs2 = pd.array(ldaDF_14.iloc[i - neighbor,4:18], dtype='float') #for the previous tweet
        # compute the novelty using KL divergence
        novelties.append(scipy.stats.entropy(probs1,probs2))
        # save the topic loadings for the subsequent tweet
        probs2 = pd.array(ldaDF_14.iloc[i + neighbor,4:18],dtype='float')
        # compute the transience using KL divergence
        transiences.append(scipy.stats.entropy(probs1,probs2))
        end = time.time() #time the loop
    # estimate the average novelty and transience for the target tweet and save it to df   
    ldaDF_14['novelty'][i] = np.mean(novelties)
    ldaDF_14['transience'][i]= np.mean(transiences)


#%% compute resonance
ldaDF_14['resonance'] = ldaDF_14['novelty'] - ldaDF_14['transience']
scores_day = ldaDF_14[['created_at', 'status_id',
                       'text','novelty','transience',
                       'resonance']]

#%% save results
scores_day.to_csv(r'.\Scores\day.csv')
