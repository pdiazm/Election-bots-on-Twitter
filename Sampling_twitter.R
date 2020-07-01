#Load libraries
library(tidyverse)
library(rtweet)
library(here)

# create  elements of query using words tweeted by bots according to Bot Sentinel
w2= '"Fake News" OR #FakeNews'
w3= "#Trump2020 OR Trump OR President OR 'President Trump'"
w4= "@realDonaldTrump OR @JoeBiden"
w5= "Biden OR 'Joe Biden'"
w6= "#WWG1WGAn OR #MAGA"

# filter tweets by english language
lg = "lang:en"

# join keywords into  a Twitter API query format
q = str_c(w2,'OR', w3,'OR', w4,'OR', w5,'OR', w6, lg, sep = " ")

# sample Twitter using the search_tweets function
rt <- search_tweets(
  # use the query created above
  q,
  # specify the number of tweets you want to collect
  n = 335299,
  # use max_id to specify the point at which the function should start collecting tweets
  # max_id = 1258359193893048321,
  # set to True to ensure the function collects tweets past the rate limit (18,000)
  retryonratelimit = T
)

# create function to clean the response dataframe
clean <- function(tweet_data){
  new <- tweet_data %>%
    #remove all non-digit values from the IDs (this is only needed when you combine the response with data that was previously stored as csv)
    mutate(user_id = str_replace_all(user_id, "\\D", ""),
           status_id = str_replace_all(status_id, "\\D", ""),
           #change timezone to ET
           created_at=with_tz(created_at, tzone = "America/New_York")) %>%
    #filter columns to keep relevant data
    select(user_id, status_id,created_at,screen_name,text,is_retweet,
           favorite_count,retweet_count,hashtags,mentions_screen_name,
           retweet_favorite_count,retweet_count,retweet_followers_count,
           retweet_friends_count,retweet_verified,description,
           followers_count,favorite_count,friends_count,statuses_count,
           account_created_at,verified,location) 
  return(new)
} 

# clean the tweet data using the above function
new_tweets <- clean(rt)

# save results
write_as_csv(new_tweets, here::here("file_name.csv"))

