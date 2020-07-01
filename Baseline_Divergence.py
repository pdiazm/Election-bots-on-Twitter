# -*- coding: utf-8 -*-
"""
Creating a baseline divergence measure

Created on Fri May 29 12:21:38 2020

@author: Paulino Diaz
"""

#%% load libraries
import lucem_illud_2020 as lucem
import pandas as pd
import numpy as np
import random

#%% load data
days = pd.read_csv(r'.\Classified\all.csv')
days = days.reset_index().drop(['index','Unnamed: 0'], axis=1)

#%% create two random samples
random.seed(23)
sample= days.sample(n=2000)
sample1= sample.sample(n=1000)
sample2 = sample.loc[~sample.status_id.isin(sample1['status_id'])]

#%% normalize text data for sample1
sample1 = sample1[['user_id','created_at','text']]
sample1['tokenized_text'] = sample1['text'].apply(lambda x: lucem.word_tokenize(x))
sample1['normalized_text'] = sample1['tokenized_text'].apply(lambda x: lucem.normalizeTokens(x,
                                                                                               extra_stop=['>',
                                                                                                           '<',
                                                                                                           'u+0001f1fa><u+0001f1f8']))  
sample1.head(3)

#%% normalize text data for sample2
sample2 = sample2[['user_id','created_at','text']]
sample2['tokenized_text'] = sample2['text'].apply(lambda x: lucem.word_tokenize(x))
sample2['normalized_text'] = sample2['tokenized_text'].apply(lambda x: lucem.normalizeTokens(x,
                                                                                               extra_stop=['>',
                                                                                                           '<',
                                                                                                           'u+0001f1fa><u+0001f1f8']))  
sample2.head(3)

#%% counting top words for sample1
sample1_countsDict = {}
for word in sample1['normalized_text'].sum():
    if word in sample1_countsDict:
        sample1_countsDict[word] += 1
    else:
        sample1_countsDict[word] = 1
sample1_word_counts = sorted(sample1_countsDict.items(), key = lambda x : x[1], reverse = True)

#%% counting top words for sample2
sample2_countsDict = {}
for word in sample2['normalized_text'].sum():
    if word in sample2_countsDict:
        sample2_countsDict[word] += 1
    else:
        sample2_countsDict[word] = 1
sample2_word_counts = sorted(sample2_countsDict.items(), key = lambda x : x[1], reverse = True)


#%% create dataframe with overlapping words and their counts
sample1_wordCount = pd.DataFrame(sample1_word_counts,columns=['words','count'])
sample2_wordCount = pd.DataFrame(sample2_word_counts,columns=['words','count'])

blol_wordCount = pd.merge(sample1_wordCount,sample2_wordCount,on='words',how='inner')
blol_wordCount['sample1_probs'] = blol_wordCount['count_x'] / blol_wordCount['count_x'].sum()
blol_wordCount['sample2_probs'] = blol_wordCount['count_y'] / blol_wordCount['count_y'].sum()

#%% calculate the divergence between the two samples and its element wise divergence
sample1_array = np.array(blol_wordCount['sample1_probs'])
sample2_array = np.array(blol_wordCount['sample2_probs'])

#total divergence
sample1_KL = scipy.stats.entropy(sample1_array, sample2_array)
sample2_KL = scipy.stats.entropy(sample2_array, sample1_array)

#element-wise divergence
blol_wordCount['sample1_kl_divergence']= scipy.special.kl_div(sample1_array, sample2_array)
overlap_wordCount= blol_wordCount.sort_values(by='sample1_kl_divergence', ascending=False)

#jensen-shannon divergence
baselineJS = scipy.spatial.distance.jensenshannon(sample1_array, sample2_array)

#%% create dataframe
baseline_divergence = pd.DataFrame({'measure':['Jensen-Shannon',
                                               'KL-one_sided',
                                               'KL-one_sided'],
                                    'value': [baselineJS,
                                              sample1_KL,
                                              sample2_KL]})
baseline_divergence.to_csv(r'.\baseline_divergence.csv')
