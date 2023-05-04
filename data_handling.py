import pymongo
import pandas as pd
from similarity import get_text_similarities
from TFIDF import get_tfidf
from collections import Counter as cn
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import datetime

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['DB_name']

def combine_cols(target_col_name, source_cols_names):
    target_col= db[target_col_name]
    for name in source_cols_names:
        print(name)
        source_col = db[name]
        source_docs = list(source_col.find())
        for d in source_docs:
            target_col.insert_one(d)

def add_interaction_field(col_name):
    col = db[col_name]
    col.aggregate([{"$addFields": {"interaction_count": {"$add": ["$favorite_count", "$retweet_count", "$qoute_count"]}}}, {"$out": col_name}])

def get_top_tweets_rt(col_name, since, until):
    col = db[col_name]
    top_tweets = list(col.find({"created_at": {"$gte": since, "$lt": until}}, {"_id": 0, "id": 1, "user_screen_name": 1, "user_display_name": 1, "created_at": 1,
                                    "full_text": 1, "favorite_count":1, "retweet_count": 1, "qoute_count": 1, "reply_count": 1,
                                    "interaction_count": 1, "tweet_link":1, }).sort("retweet_count", -1).limit(50))
    df = pd.DataFrame.from_records(top_tweets)

    df.to_excel("Analysis_2/top_50_tweets_" + col_name + "_retweets.xlsx")

def get_top_tweets_fav(col_name, since, until):
    col = db[col_name]
    top_tweets = list(col.find({"created_at": {"$gte": since, "$lt": until}}, {"_id": 0, "id": 1, "user_screen_name": 1, "user_display_name": 1, "created_at": 1,
                                    "full_text": 1, "favorite_count":1, "retweet_count": 1, "qoute_count": 1, "reply_count": 1,
                                    "interaction_count": 1, "tweet_link":1, }).sort("favorite_count", -1).limit(50))
    df = pd.DataFrame.from_records(top_tweets)

    df.to_excel("Analysis_2/top_50_tweets_" + col_name + "_favorites.xlsx")

def get_top_tweets_intr(col_name, since, until):
    col = db[col_name]
    top_tweets = list(col.find({"created_at": {"$gte": since, "$lt": until}}, {"_id": 0, "id": 1, "user_screen_name": 1, "user_display_name": 1, "created_at": 1,
                                    "full_text": 1, "favorite_count":1, "retweet_count": 1, "qoute_count": 1, "reply_count": 1,
                                    "interaction_count": 1, "tweet_link":1, }).sort("interaction_count", -1).limit(50))
    df = pd.DataFrame.from_records(top_tweets)

    df.to_excel("Analysis_2/top_50_tweets_" + col_name + "_interaction.xlsx")

def remove_duplicates_form_col(col_name):
    col = db[col_name]
    cursor = col.aggregate(
        [
            {"$group": {"_id": "$id", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
            {"$match": {"count": {"$gte": 2}}}
        ]
    )
    response = []
    for doc in cursor:
        del doc["unique_ids"][0]
        for id in doc["unique_ids"]:
            response.append(id)
    col.delete_many({"_id": {"$in": response}})


def get_text_similarities_from_col(col_name, since, until):
    print(since)
    print(until)
    col = db[col_name]
    result = list(col.find({"created_at": {"$gte": since, "$lt": until}}))
    print(len(result))
    result = [x["full_text"] for x in result]
    df = get_text_similarities(result)
    df.to_excel("Analysis_2/similarities/text_similarities_"+ col_name + ".xlsx")


def export_excel_from_tfidf(dir, col_name):
    col = db[col_name]
    result = col.find({})
    print(result)
    df = pd.DataFrame(list(result))
    df.to_excel(dir + col_name + ".xlsx")

def get_tfidf_from_col(col_name, dir, lang, since, until):
    get_tfidf(col_name, lang, since, until)
    export_excel_from_tfidf(dir, "tfidf_" + col_name +"_2")

def get_words_frequency(col_name):
    col = db[col_name]
    result = list(col.find())
    result = [x["full_text"] for x in result]
    total_text = ""
    for t in result:
        total_text = total_text + t + " "

    filtered_text = clean_text(total_text)
    counter = cn(filtered_text)
    most_occur = counter.most_common(10)
    print(most_occur)

def get_words_frequency_bigram(col_name):
    col = db[col_name]
    result = list(col.find())
    result = [x["full_text"] for x in result]
    total_text = ""
    for t in result:
        total_text = total_text + t + " "
    words_list = clean_text(total_text)
    ngrams = zip(words_list, words_list[1:])
    counter = cn(ngrams)
    most_occur = counter.most_common(10)
    print(most_occur)

def get_words_frequency_trigram(col_name):
    col = db[col_name]
    result = list(col.find())
    result = [x["full_text"] for x in result]
    total_text = ""
    for t in result:
        total_text = total_text + t + " "
    words_list = clean_text(total_text)
    ngrams = zip(words_list, words_list[1:], words_list[2:])
    counter = cn(ngrams)
    most_occur = counter.most_common(10)
    print(most_occur)

def get_words_frequency_quadgram(col_name):
    col = db[col_name]
    result = list(col.find())
    result = [x["full_text"] for x in result]
    total_text = ""
    for t in result:
        total_text = total_text + t + " "
    words_list = clean_text(total_text)
    ngrams = zip(words_list, words_list[1:], words_list[2:], words_list[3:])
    counter = cn(ngrams)
    most_occur = counter.most_common(10)
    print(most_occur)

def clean_text(text):
    new_text = re.sub(r"http\S+", "", text)
    new_text = re.sub("@\S+", "", new_text)
    new_text = new_text.replace("?", "")
    new_text = new_text.replace("؟", "")
    new_text = new_text.replace("!", "")
    new_text = new_text.replace("#", "")
    new_text = new_text.replace(".", "")
    new_text = new_text.replace(":", "")
    new_text = new_text.replace(" اللي ", " ")

    stop_words = set(stopwords.words('english') + stopwords.words('arabic') + stopwords.words('spanish'))

    word_tokens = word_tokenize(new_text)
    filtered_sentence = [w for w in word_tokens if (not w.lower() in stop_words) and w!="''" and w!="``"]
    return filtered_sentence
