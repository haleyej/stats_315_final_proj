# Code for final project for STATS 315
# Cleans and preprocesses text
# Haley Johnson

import pandas as pd
from langdetect import DetectorFactory, detect_langs, detect
import emoji

def get_lang(s):
    DetectorFactory.seed = 42
    try:
        return detect_langs(s)
    except:
        return []

def remove_emojis(s):
    valid = ''
    for i in range(len(s)):
        c = s[i]
        if emoji.is_emoji(c) == False:
            valid += c
    return valid

def clean_text(df):
  # filter out non-english tweets
    df['message_lang'] = df.message.apply(get_lang)
    df.message_lang = df.message_lang.apply(lambda s: [l.lang for l in s if l.lang == 'en' and l.prob >= 0.90 and type(l) != 'str'])
    df = df[df.message_lang.apply(lambda s: 'en' in s)]

    #remove emojis
    df['message_cleaned'] = df.message.apply(remove_emojis)

    #remove non-ASCII characters
    df.message_cleaned = df.message_cleaned.apply(lambda s: s.encode('ascii', errors='ignore').decode())

    # remove retweet and reply markers
    df.message_cleaned = df.message_cleaned.str.replace("RT\s@\S+?:\s", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("^(@\S+\s)+?", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("@", "", regex = True)
    #truncate links
    df.message_cleaned = df.message_cleaned.str.replace('https?:\S+', 'https', regex = True)

    #remove hashtags 
    df.message_cleaned = df.message_cleaned.str.replace("#", "", regex = True)

    #get rid of quotations 
    df.message_cleaned = df.message_cleaned.str.replace("'", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace('"', '', regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("''", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace('‘', "", regex = True)

    #more punctuation 
    df.message_cleaned = df.message_cleaned.str.replace("…", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace('.', "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace('�', "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("?", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("!", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("&amp", " ", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace(":", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace(";", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("-", " ", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("–", " ", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("$", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace("(", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace(")", "", regex = True)
    df.message_cleaned = df.message_cleaned.str.replace(",", "", regex = True)

    #fix white space
    df.message_cleaned = df.message_cleaned.str.replace('\s\s+', ' ', regex = True)
    df.message_cleaned = df.message_cleaned.str.lstrip()

    df['message_cleaned'] = df['message_cleaned'].astype('str')

    return df 

def main():
    df = pd.read_csv("data/twitter_sentiment.csv")
    df = df.dropna(subset=['message', 'sentiment'])

    df.sentiment = df.sentiment + 2

    df = clean_text(df)
    df.to_csv("data/cleaned_text.csv")

if __name__ == "__main__":
    main()