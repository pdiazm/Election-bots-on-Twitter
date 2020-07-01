# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:03:14 2020

@author: Paulino Diaz
"""
#%% load libraries
import json
import botometer
import pandas as pd
import random

#%% get and save your Twitter API and Botometer credentials

# specify path in which credentials are saved
twitter_path = r"C:\Users\pauli\OneDrive\Documents\Creds\creds-twitter"
rapid_path = r"C:\Users\pauli\OneDrive\Documents\Creds\creds-rapid"

# open json files where credentials are stored and save contents
with open(twitter_path) as f:
    t_creds = json.load(f)
    
with open(rapid_path) as f:
    r_creds = json.load(f)

# save keys into dictionary for easy access
rapidapi_key = r_creds['rapidapi_key']
twitter_app_auth = {
    'consumer_key': t_creds['consumer_key'],
    'consumer_secret': t_creds['consumer_secret'],
    'access_token': t_creds['access_token'],
    'access_token_secret': t_creds['access_token_secret'],
  }

# initialize the botometer account
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)

#%% define functions to send requests to Botometer

#defining chunks to accomodate max requests per 15 min window
def chunks(lst, n):
    #yield succesive chunks of length n
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
 
# define function to query classifications from botometer   
def get_bots(chunk, errors):
    n_errors = errors
    for screen_name, result in bom.check_accounts_in(chunk):
        try:
            user = [screen_name, 
                    result['cap']['english']]
            results.append(user)
        except KeyError:
            n_errors += 1
            pass
    return results, n_errors

#%% load tweets (in this case those sampled on Friday) and pre-process for classification

tweets = pd.read_csv(r'.\days\friday.csv')

# keep only unique accounts
accounts = tweets.iloc[:,0].drop_duplicates()
random.seed(6)
# n = the max number of requests per day accepted by Botometer
accounts = accounts.sample(n=17280) 

# create various chunks of accounts to loop over
# 180 is the max number of requests the Botometer API can receive per query
account_chunks = list(chunks(accounts,180))

# loop over all user_ids to classify the accounts in the sample
results =[]
errors=0
for chunk in account_chunks:
    get_bots(chunk, errors)
    
#saved calssified results
fri = pd.DataFrame(results, columns=['user_id','cap'])
fri = pd.merge(tweets,fri,on='user_id', how='inner')
fri.to_csv(r'.\Classified\friday.csv')



