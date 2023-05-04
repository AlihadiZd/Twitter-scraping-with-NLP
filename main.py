import snscrape.modules.twitter as sntwitter
import pandas as pd
import tkinter as tk
import datetime
from getmac import get_mac_address as gmac
import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['DB_name']

def validate_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        print("Incorrect data format, should be YYYY-MM-DD")
        return False

def get_tweets(filename,phrase="", from_user="" ,since="", until=""):
    attributes_container = []
    q = ''
    if phrase != '':
        q = phrase
        print(q)
    if from_user != "":
        q = q + " (from:"+ from_user + ")"
    if until != "":
        q = q + " until:" + until
    if  since != "":
        q = q + " since:" + since

    temp_dict = {}
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(q).get_items()):
        temp_dict = {}
        if i % 100 == 0:
            print(i)

        temp_dict = {"id" : str(tweet.id), "user_screen_name" : tweet.user.username, "user_display_name": tweet.user.displayname,
                    "created_at": tweet.date.replace(tzinfo=None), "full_text": tweet.content, "user_id": str(tweet.user.id), "favorite_count": tweet.likeCount,
                    "retweet_count": tweet.retweetCount, "qoute_count": tweet.quoteCount, "reply_count": tweet.replyCount, "lang": tweet.lang,
                    "tweet_link": tweet.url, "hashtags": tweet.hashtags, "outlinks": tweet.outlinks,
                    "in_reply_to_tweet_id": str(tweet.inReplyToTweetId),
                    "cashtags": tweet.cashtags, "conversation_id": str(tweet.conversationId), "source": tweet.source,
                    "user_friends_count": tweet.user.friendsCount, "user_followers_count": tweet.user.followersCount}

        db[filename].insert_one(temp_dict)

        attributes_container.append(
            [str(tweet.id), tweet.user.username, tweet.user.displayname, tweet.date, tweet.content, str(tweet.user.id), tweet.likeCount,
             tweet.retweetCount, tweet.quoteCount, tweet.replyCount, tweet.lang, tweet.url, tweet.hashtags, tweet.outlinks,
             str(tweet.inReplyToTweetId),
             tweet.cashtags, str(tweet.conversationId), tweet.source, tweet.user.friendsCount, tweet.user.followersCount])

    tweets_df = pd.DataFrame(attributes_container,
                             columns=["id", "user_screen_name", "user_display_name", "created_at", "full_text", "user_id", "favorite_count",
                                      "retweet_count", "qoute_count", "reply_count", "lang", "tweet_link", "hashtags", "outlinks",
                                      "in_reply_to_tweet_id",
                                      "cashtags", "conversation_id", "source", "user_friends_count", "user_followers_count"])

    if not tweets_df.empty:
        tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize(None)

    tweets_df.to_excel(filename+".xlsx")

def get_formated_error_text(error_text):
    error_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_text = str(error_date) + "\n\n" + error_text + "\n\n" + "-------------------" + "\n\n"
    return total_text

def append_error(error_text):
    f = open("errors.txt", "a")
    text = get_formated_error_text(error_text)
    f.write(text)
    f.close()

def create_window():
    root = tk.Tk()
    root.geometry("400x300")
    root.title("Twitter Data Gathering")

    def Take_input():
        keyword = inputtxt.get("1.0", tk.END).replace("\n","")
        from_user = inputUser.get("1.0", tk.END).replace("\n","")
        since = inputDateSince.get("1.0", tk.END).replace("\n","")
        until = inputDateUntil.get("1.0", tk.END).replace("\n","")

        if since == "":
            append_error("Empty Since field")
            return
        else:
            if not validate_date(since):
                append_error("Invalid Since date format")
                return
        if until != "":
            if not validate_date(until):
                append_error("Invalid Until date format")
                return
        else:
            until = datetime.datetime.now() + datetime.timedelta(days=1)
            until = until.strftime("%Y-%m-%d")

        filename = outputFilename.get("1.0", tk.END).replace("\n","")
        if filename == "":
            append_error("Empty Output File Name field")
            return
        if keyword!= "" or from_user != "":
            get_tweets(filename, keyword, from_user, since, until)
        else:
            append_error("Both 'Keyword(s)' and 'From User' fields are empty\nAt least one of them should be filled")

    text_label = tk.Label(text="Keyword(s)")
    user_label = tk.Label(text="From User")
    since_label = tk.Label(text="Since")
    until_label = tk.Label(text="Until")
    filename_label = tk.Label(text="Output file name")
    spacing_label = tk.Label(text="")
    inputtxt = tk.Text(root, height=1,
                    width=35,
                    bg="light yellow")

    inputUser = tk.Text(root, height=1,
                       width=35,
                       bg="light yellow")

    inputDateSince = tk.Text(root, height=1,
                       width=20,
                       bg="light yellow")

    inputDateUntil = tk.Text(root, height=1,
                       width=20,
                       bg="light yellow")

    outputFilename = tk.Text(root, height=1,
                             width=20,
                             bg="light yellow")

    Display = tk.Button(root, height=1,
                     width=20,
                     text="Submit",
                     command=lambda: Take_input())


    text_label.pack()
    inputtxt.pack()
    user_label.pack()
    inputUser.pack()
    since_label.pack()
    inputDateSince.pack()
    until_label.pack()
    inputDateUntil.pack()
    filename_label.pack()
    outputFilename.pack()
    spacing_label.pack()
    Display.pack()
    tk.mainloop()


try:
    create_window()
except BaseException:
    append_error(BaseException)
