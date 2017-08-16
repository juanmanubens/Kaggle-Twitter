#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:55:45 2017

@author: Juan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
import textmining as txt
from wordcloud import WordCloud, STOPWORDS
from operator import itemgetter
from PIL import Image
import webcolors
from sklearn.ensemble import RandomForestClassifier
# Using vectorizer from kaggle
from sklearn.cross_validation import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer

### Step 1 - Data Preview and Cleaning ###

# read data
df = pd.read_csv('twitter.csv', encoding='latin1')
df.describe()

# get dimensions and columns
(nrows,ncols) = df.shape
cols = set(df.columns)

"""
Index(['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
       '_last_judgment_at', 'gender', 'gender:confidence', 'profile_yn',
       'profile_yn:confidence', 'created', 'description', 'fav_number',
       'gender_gold', 'link_color', 'name', 'profile_yn_gold', 'profileimage',
       'retweet_count', 'sidebar_color', 'text', 'tweet_coord', 'tweet_count',
       'tweet_created', 'tweet_id', 'tweet_location', 'user_timezone'],
      dtype='object')
"""
# get data types of columns
df.dtypes 
"""
_unit_id                   int64
_golden                     bool
_unit_state               object
_trusted_judgments         int64
_last_judgment_at         object
gender                    object
gender:confidence        float64
profile_yn                object
profile_yn:confidence    float64
created                   object
description               object
fav_number                 int64
gender_gold               object
link_color                object
name                      object
profile_yn_gold           object
profileimage              object
retweet_count              int64
sidebar_color             object
text                      object
tweet_coord               object
tweet_count                int64
tweet_created             object
tweet_id                 float64
tweet_location            object
user_timezone             object

need to change _unit_id to categorical, gender to categorical, 
created to date, and tweet_created to date
"""
# convert data types
for col in ['_unit_id','gender','link_color','sidebar_color']:
    df[col] = df[col].astype('category')

def toDate(s):
    return s.split()[0]

for col in ['created','tweet_created']:
    df[col] = pd.to_datetime(df[col].apply(toDate),format = '%m/%d/%Y')

# get unique values for gender columns and their counts and remove nan's
df.apply(pd.isnull).sum()
df = df[pd.notnull(df.gender)]
df.description = df.description.fillna("")

df.gender.value_counts(dropna=False)

# remove rows with unknown and brand for gender
df = df[df.gender != 'unknown']
df = df[df.gender != 'brand']
df.gender = df.gender.cat.remove_unused_categories()

# normalize text of tweets and descriptions 
def normalize_text(s):
    s = str(s)
    s = s.lower()    
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    s = re.sub('\s+',' ',s)   
    return s

    
df.text = df.text.apply(normalize_text)
df.description = df.description.apply(normalize_text)
    
# proportion of observations with gender confidence < 1
conf = df.loc[:,'gender:confidence'] >= 1
(nrows,_) = df.shape
sum(conf)/nrows 
df = df[df['gender:confidence'] >= 1]


# remove columns and rows which we will not use
toKeep = cols - set(['_golden', '_unit_state', '_trusted_judgments', '_last_judgment_at', 'profile_yn', 'profile_yn:confidence','gender_gold','profile_yn_gold', 'profileimage','tweet_coord','user_timezone','tweet_location','gender:confidence'])

# remove columns 
clean = df[list(toKeep)]

# check for duplicated users
sum(clean.duplicated('_unit_id'))

# summary of missing values
len(clean.index)-clean.count()

# show preview
clean.head()
(nrows,ncols) = clean.shape



### Step 2 - Feature Creation ###

"""
From the cleaned data set, there are many features that we can create.

1. has_mention --> When tweeting, a person has the option
of tagging another user in the tweet. This is done by writing '@' and 
the other user's twitter name without any spaces. A twitter name can be a 
maximum of 15 characters so we will look for ocurrences of '@' followed by
a max of 15 characters. To avoid counting ocurrences that fulfill these requirements 
but are not mentions, we note that twitter handles only allow alphanumeric characters
and underscores. More info at: https://support.twitter.com/articles/101299.

test = re.compile(r'(^|[^@\w])@(\w{1,15})\b', re.IGNORECASE)
sample = r'@giannaaa28 lmao _Ù÷â_Ù÷â dude Im hella scared for next episode bc the ending to yesterdays'
re.findall(test,sample)

2. days_active --> days since the account was created until the tweet was created

3. tweets_per_day --> how many tweets the user has written from the moment the account was 
created to when the tweet was collected

4. likely_fake --> binary variable indicating whether or not the account could be a bot
based on the ridiculous number of tweets per day

5. description_nchars --> indicates the number of characters in the user's description

6. text_nchars --> indicates the number of characters in the tweet

7. name_nchars --> indicates the number of characters in the user's name

8.has_hashtags --> indictes the number of hashtags in the tweet

"""

# has_mentions (binary indicating if tweet has a mention or not)

def countMentions(s):  
        mention = re.compile(r'(^|[^@\w])@(\w{1,15})\b', re.IGNORECASE)
        found = mention.findall(s)
        return (len(found) >= 1)*1
 
clean['has_mention']  = clean.loc[:,'text'].apply(countMentions)

# days_active

clean['days_active'] = ((clean.tweet_created - clean.created) / np.timedelta64(1, 'D'))+1

#clean.groupby('gender').days_active.plot(kind='bar')

# tweets_per_day 

clean['tweets_per_day'] = clean['tweet_count'] / clean['days_active']

# favorites per day

clean['favorites_per_day'] = clean['fav_number'] / clean['days_active']

# retweets per day 

clean['retweets_per_day'] = clean['retweet_count'] / clean['days_active']

# remove observations with more than 75 tweets per day and more than 200 favorite per day

clean = clean.loc[clean['tweets_per_day'] <=75,:]
clean = clean.loc[clean['favorites_per_day'] <= 200,:]
# re index series
clean = clean.reset_index(drop=True)

# decripton nchars (count whitespace??)

clean['description_nchars'] = clean.description.apply(len)
    

# text nchars

clean['text_nchars'] = clean.text.apply(len)


# name nchars

clean['name_nchars'] = clean.name.apply(len)

# has_hashtag

def countHashTags(s):  
        mention = re.compile(r'^#(\w{1,15})\b', re.IGNORECASE)
        found = mention.findall(s)
        return (len(found) >= 1)*1

clean['has_hashtag']  = clean.text.apply(countHashTags)

clean = clean.reset_index(drop=True)


# link_color_name  and sidebar_color_name
# (import webcolors (http://webcolors.readthedocs.io/en/1.7/install.html))
# Have to get closest color, not exact one, got these functions from the internet - http://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green. colors are in html4 format which only support 16 colors
# conversion is to the most smilar color, not the exact one. 

def closest_colour(requested_colour):
    min_colours = {}
    new = webcolors.hex_to_rgb(requested_colour)
    for key, name in webcolors.html4_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - new[0]) ** 2
        gd = (g_c - new[1]) ** 2
        bd = (b_c - new[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    new = webcolors.hex_to_rgb(requested_colour)
    try:
        closest_name = webcolors.rgb_to_name(new,spec='html4')
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name

def convert(hex):
    if len(hex) == 6:
        return get_colour_name('#'+hex)
    else:
        return 'No color specified'

clean['link_color_name'] = clean.link_color.apply(convert)
clean['link_color_name'] = clean['link_color_name'].astype('category')
clean.link_color_name.value_counts(dropna=False)

clean['sidebar_color_name'] = clean.sidebar_color.apply(convert)
clean['sidebar_color_name'] = clean['sidebar_color_name'].astype('category')
clean.sidebar_color_name.value_counts(dropna=False)



### Step 3 - Exploring and Visualizing the Data ###

# plot proportions of males and females

plt.figure();
ax = clean.gender.value_counts().plot(kind='pie')
ax.set_xlabel('test')
plt.show()


# Plot tweets per day by gender
# omitting values greater than 150
omit = clean.loc[clean['tweets_per_day']>=80,'tweets_per_day']
len(omit) # 315 observations

fig = plt.figure()
gs=GridSpec(2,2)

xm = clean.loc[clean['gender']=='male','tweets_per_day']
xf = clean.loc[clean['gender']=='female','tweets_per_day']
bins = np.linspace(0,100,25)

ax = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

ax.hist(xm,bins,alpha=0.6,color='c')
ax.hist(xf,bins,alpha=0.6,color='g')
ax.tick_params(labelsize=20,pad=20)
ax.set_xticks(list(range(0,100,10)))
ax.set_xlim([0,100])
ax.legend(loc='upper right',labels=['Males','Females'],fontsize=30)
t = ax.set_title('Tweets per day by gender',fontsize=30)
t.set_y(1.05)

ax2.hist(xm,bins,color='c')
ax2.tick_params(labelsize=15,pad=20)
ax2.set_xticks(list(range(0,100,10)))
ax2.set_xlim([0,100])
t2 = ax2.set_title('Tweets per day - males',fontsize=20)
t2.set_y(1.05)

ax3.hist(xf,bins,color='g')
ax3.tick_params(labelsize=15,pad=20)
ax3.set_xticks(list(range(0,100,10)))
ax3.set_xlim([0,100])
ax3.set_ylim([0,1800])
t3 = ax3.set_title('Tweets per day - females',fontsize=20)
t3.set_y(1.05)

fig.tight_layout()
fig.show()



"""
To-do: 
    - Create same plots but for females and males separately
    - Create the same plots for description separately
    - Input word term frequences into data frame
"""

# remove stopwords from tweets and descriptions
# for testing, I joined the descriptions and tweets into one string
d = np.array([clean._unit_id,clean.text,clean.description,clean.gender]).T
words = pd.DataFrame(d, columns=['_unit_id','text','description','gender'])
words['clean_text'] = words.text + words.description
words['clean_text'] = words.clean_text.apply(txt.simple_tokenize_remove_stopwords)




# create separate data frames for males and females
df_males = words.loc[words['gender']=='male',:]
df_males = df_males.reset_index(drop=True)
df_females = words.loc[words['gender']=='female',:]
df_females = df_females.reset_index(drop=True)

# get list of all the words 
allwordstweets = []
(nrows,_) = words.shape
for i in range(nrows):
    allwordstweets += words.loc[i,'clean_text']

allwordstweets = [word for word in allwordstweets if 'Ù' not in word]
allwordstweets = [word for word in allwordstweets if 'http' not in word]
allwordstweets = [word for word in allwordstweets if 'â' not in word]

# remove duplicated words
allwordstweets2 = list(set(allwordstweets))
len(allwordstweets2) # 36927 repeated words (not including stopwords)

# create list tuple with words repeated 100 times or more
toReturn = []
for word in allwordstweets2:
    count = allwordstweets.count(word)
    if  count >= 100:
        toReturn.append((word, count))
    else:
        pass

# sort the words by their counts (descending order)
sortedbycount = sorted(toReturn,key=itemgetter(1),reverse=True)
sortedbycount
# get words and their counts separately
x = [x for (x,y) in sortedbycount[0:25]]
y = [y for (x,y) in sortedbycount[0:25]]

### same process but separated by gender ###

allwordstweets_males = []
(nrows_m,_) = df_males.shape
for i in range(nrows_m):
    allwordstweets_males += df_males.loc[i,'clean_text']

allwordstweets_males = [word for word in allwordstweets_males if 'Ù' not in word]
allwordstweets_males = [word for word in allwordstweets_males if 'http' not in word]
allwordstweets_males = [word for word in allwordstweets_males if 'â' not in word]

# remove duplicated words
allwordstweets_males2 = list(set(allwordstweets_males))
len(allwordstweets_males2) # 
# create list tuple with words repeated 100 times or more
toReturn_males = []
for word in allwordstweets_males2:
    count = allwordstweets_males.count(word)
    if  count >= 50:
        toReturn_males.append((word, count))
    else:
        pass

# sort the words by their counts (descending order)
sortedbycount_males = sorted(toReturn_males,key=itemgetter(1),reverse=True)
sortedbycount_males

# get words and their counts separately
x_m = [x for (x,y) in sortedbycount_males[0:25]]
y_m = [y for (x,y) in sortedbycount_males[0:25]]



# females
 
### same process but separated by gender ###

allwordstweets_females = []
(nrows_f,_) = df_females.shape
for i in range(nrows_f):
    allwordstweets_females += df_females.loc[i,'clean_text']

allwordstweets_females = [word for word in allwordstweets_females if 'Ù' not in word]
allwordstweets_females = [word for word in allwordstweets_females if 'http' not in word]
allwordstweets_females = [word for word in allwordstweets_females if 'â' not in word]

# remove duplicated words
allwordstweets_females2 = list(set(allwordstweets_females))
len(allwordstweets_females2) # 
# create list tuple with words repeated 50 times or more
toReturn_females = []
for word in allwordstweets_females2:
    count = allwordstweets_females.count(word)
    if  count >= 50:
        toReturn_females.append((word, count))
    else:
        pass

# sort the words by their counts (descending order)
sortedbycount_females = sorted(toReturn_females,key=itemgetter(1),reverse=True)
sortedbycount_females

# get words and their counts separately
x_f = [x for (x,y) in sortedbycount_females[0:25]]
y_f = [y for (x,y) in sortedbycount_females[0:25]]

       
### Step 4 - Extracting clean dataset ###
# 
       
words['clean_text'] = words.clean_text.apply(' '.join)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(words['clean_text'])

new = pd.DataFrame(x.A, columns=vectorizer.get_feature_names())
x2 = new.loc[:,(new.sum() >= 100)]

pd.concat([words,x2])

t = pd.merge(clean,x2,right_index=True,left_index=True)

t.drop(t.columns[[0,1,2,3,4,5,7,8,9,10,11,12]],axis=1,inplace=True)
t.to_csv('vectorizerdata.csv')


       

