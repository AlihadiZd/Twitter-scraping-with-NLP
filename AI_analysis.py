import nltk
import pymongo
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline, AutoTokenizer, AutoConfig,AutoModelForSequenceClassification
import datetime
import matplotlib.pyplot as plt
import time
import pandas as pd
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
from nltk.corpus import stopwords
import re
from PIL import Image
import numpy as np

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['Freelance_messi']

def get_emotions_from_col(col_name, since, until):
    tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

    emotion = pipeline('sentiment-analysis',
                       model='arpanghoshal/EmoRoBERTa')

    col = db[col_name]
    top_tweets = list(col.find({"t_text": {"$exists": True}, "emotion": {"$exists": False},"created_at": {"$gte": since, "$lt": until}}, {"_id": 1, "created_at": 1,
                                    "full_text": 1, "t_text":1, "interaction_count":1 }).sort("interaction_count", -1))
    for t in top_tweets:
        print("\n\n")
        print(t["full_text"])
        print(t["t_text"])
        try:
            emotion_labels = emotion(t["t_text"])
            print(emotion_labels[0])
            col.update_one({"_id": t["_id"]},{"$set": {"emotion": emotion_labels[0]["label"]}})
        except BaseException as e:
            print(e)
        print("-----------------------")

def get_sentiment_from_col(col_name, since, until):
    sentiment_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    flag =1
    while flag:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(sentiment_MODEL)
            sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_MODEL)
            config = AutoConfig.from_pretrained(sentiment_MODEL)
            flag =0
        except BaseException as e:
            print(e)
            time.sleep(3)

    sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=sentiment_tokenizer)
    result = sentiment_task("Covid cases are increasing fast!")
    print(result)
    print(result[0]["label"])

    col = db[col_name]
    top_tweets = list(col.find({"t_text": {"$exists": True}, "sentiment": {"$exists": False},"created_at": {"$gte": since, "$lt": until}}, {"_id": 1, "created_at": 1,
                                    "full_text": 1, "t_text":1, "interaction_count":1 }).sort("interaction_count", -1))
    for t in top_tweets:
        print("\n\n")
        print(t["full_text"])
        print(t["t_text"])
        try:
            sentiment_labels = sentiment_task(t["t_text"])
            print(sentiment_labels)
            col.update_one({"_id": t["_id"]},{"$set": {"sentiment": sentiment_labels[0]["label"]}})
        except BaseException as e:
            print(e)
        print("-----------------------")

def piechart_for_specific_field(col_name, field_name, exceptions, since, until, title):
    col = db[col_name]
    total = col.count_documents({"created_at": {"$gte": since, "$lt": until}, "emotion": {"$exists":True,"$nin": exceptions}})
    stats= list(col.aggregate([{"$match": {"created_at": {"$gte": since, "$lt": until}, field_name: {"$exists":True,"$nin": exceptions}}},
                               {"$group": {"_id": "$"+ field_name, "count":{"$sum":1}}},{"$project":{"_id":1,"count":1,"pr":{"$divide": ["$count", total]}}},
                               {"$match": {"pr": {"$gte":0.02}}},{"$sort":{"count":-1}}]))

    def pie_nbs(pr, total):
        q = int(round(total*pr/100))
        return "{:.1f}%\n({:d})".format(pr,q)

    print(stats)
    labels = [x["_id"] for x in stats]
    print(labels)
    sizes = [x["count"] for x in stats]
    print(sizes)
    new_total = sum(sizes)
    percentages = [x["count"]/new_total for x in stats]
    explode = [0.15 if x< 0.1 else 0.02 for x in percentages]
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(8,5)
    wedges, texts, autotexts=  ax1.pie(sizes, autopct=lambda p: pie_nbs(p,new_total),
            startangle=90, explode=explode, pctdistance=1.15)

    ax1.legend(wedges, labels,
              title="Emotions",
              loc="center left",
              bbox_to_anchor=(0.9, 0, 0.5, 1))

    #plt.setp(autotexts, size=8, weight="bold")
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(title, loc='left')
    plt.savefig("AI_Analysis/"+"pie charts/"+col_name+"/"+field_name+"/"+title+".jpg")
    plt.close()
    #plt.show()
    df = pd.DataFrame(stats)
    df.to_excel("AI_Analysis/"+"pie charts/"+col_name+"/"+field_name+"/"+title+".xlsx")

def graph_for_specific_field(col_name, field_name, exceptions, since, until):
    col = db[col_name]
    top_emotions = list(col.aggregate([{"$match": {"created_at": {"$gte": since, "$lt": until}}} ,{"$group": {"_id": "$emotion", "count": {"$sum": 1}}},
                                  {"$sort": {"count":-1}},{"$match":{"_id":{"$nin": exceptions}}}, {"$limit":5}]))
    print(top_emotions)
    top_emotions_list = [x["_id"] for x in top_emotions]
    stats= list(col.aggregate([{"$match": {"created_at": {"$gte": since, "$lt": until}, field_name: {"$exists":True,"$in": top_emotions_list}}},
                               {"$project": {"date": {"$substr":["$created_at", 0, 10]}, field_name: "$"+field_name}},
                               {"$group": {"_id": {"date":"$date", field_name: "$"+field_name}, "count":{"$sum":1}}},
                               {"$sort": {"_id":1}},{"$project": {"_id": 0, "count":1,"date": "$_id.date", field_name: "$_id."+field_name}}]))
    df = pd.DataFrame(stats)
    total = col.count_documents({})
    X = ["2022-11-21"]
    X = X+ list(set([x["date"] for x in stats]))
    X.append("2022-12-20")
    emotions_list = list(set([x[field_name] for x in stats]))[:5]
    X.sort()
    print(X)
    fig1, ax1 = plt.subplots()
    for e in emotions_list:
        Y = df.loc[df[field_name] == e]["count"].tolist()
        X_emotion = df.loc[df[field_name] == e]["date"].tolist()
        docs_list = []
        for i in range(len(X_emotion)):
            doc = {"date": X_emotion[i], "count":Y[i]}
            docs_list.append(doc)
        new_Y = [df[(df[field_name]==e) & (df["date"] ==y)]["count"].iloc[0] if not df[(df[field_name]==e) & (df["date"] ==y)].empty else 0 for y in X]

        plt.plot(X, new_Y, label=e)

    plt.tick_params(axis='x', labelrotation=90)
    plt.legend()
    fig1.set_size_inches(12,6)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("AI_Analysis/graph_" + col_name + ".jpg")
    plt.show()

def frequency_comparative_graph(col_names, since, until):
    list_2 = []
    for col_name in col_names:
        print(col_name)
        temp_col = db[col_name]
        stats= list(temp_col.aggregate([{"$match": {"created_at": {"$gte": since, "$lt": until}}},
                                        {"$project": {"date": {"$substr":["$created_at", 0, 13]}}},
                                        {"$group": {"_id": {"date":"$date"}, "count":{"$sum":1}}},
                                        {"$sort": {"_id":1}},{"$project": {"_id": 0, "count":1,"date": { "$replaceAll": {"input": "$_id.date", "find": "T", "replacement": " " }}}}]))
        list_2.append(stats)
    for l in list_2:
        print(l)
    X = list(set([x["date"].replace("T", " ") for x in list_2[0]]))
    X.sort()
    fig1, ax1 = plt.subplots()
    c = 0
    for e in list_2:
        df = pd.DataFrame(e)
        Y = df["count"].tolist()
        X_dates = df["date"].tolist()
        X_dates = [x.replace("T", " ") for x in X_dates]
        docs_list = []
        for i in range(len(X_dates)):
            doc = {"date": X_dates[i], "count":Y[i]}
            docs_list.append(doc)
        new_Y = [df[df["date"] ==y]["count"].iloc[0] if not df[df["date"] ==y].empty else 0 for y in X]

        plt.plot(X, new_Y, label=col_names[c])
        c+=1
    plt.tick_params(axis='x', labelrotation=90)
    plt.legend()
    fig1.set_size_inches(12,6)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("AI_Analysis/graph_" + "comparison_2" + ".jpg")
    plt.show()

def copy_ai_fields_to_another_col(source_col_name, target_col_name):
    source_col = db[source_col_name]
    target_col = db[target_col_name]
    docs = list(source_col.find({"emotion": {"$exists": True}, "sentiment": {"$exists": True}}))
    s =0
    for d in docs:
        target_col.update_one({"id": d["id"]},{"$set": {"emotion": d["emotion"], "sentiment": d["sentiment"]}})
        s+=1
        print(d)
        print(s)

def create_word_cloud_for_top_emotions(col_name, since, until):
    col = db[col_name]
    result = col.aggregate([{"$match": {"created_at": {"$gte": since, "$lt": until},"emotion": {"$nin": ["neutral", "curiosity"]}}},{"$group": {"_id": "$emotion", "count": {"$sum":1}}}, {"$sort": {"count": -1}}, {"$limit": 3}])

    arabic_stop_words = stopwords.words('arabic')
    arabic_stop_words.append("وهذا")
    arabic_stop_words.append("وفي")
    arabic_stop_words.append("من")
    arabic_stop_words.append("وانما")
    arabic_stop_words.append("الى")
    arabic_stop_words.append("بكل")
    arabic_stop_words.append("ان")
    arabic_stop_words.append("عندما")

    for r in result:
        print(r)
        total_text = ""
        emotion_result = list(col.find({"created_at": {"$gte": since, "$lt": until}, "emotion": r["_id"]}))
        print(len(emotion_result))
        for d in emotion_result:
            total_text += d["t_text"] + " "

        total_text = re.sub(r"http\S+", "", total_text)
        total_text = total_text.replace("_", " ")
        total_text = total_text.replace("ميسي وينه", "")
        total_text = total_text.replace("السعودية", "")
        total_text = total_text.replace("السعوديه", "")
        total_text = total_text.replace("الارجنتين", "")
        total_text = total_text.replace("الأرجنتين", "")
        total_text = total_text.replace("Messi Winah", "")
        total_text = total_text.replace("Where Is Messi", "")
        total_text = total_text.replace("where is messi", "")
        total_text = total_text.replace("Where is messi", "")
        total_text = total_text.replace("Where is Messi", "")
        if col_name == "messi_es":
            total_text = total_text.replace("Dónde", "")
            total_text = total_text.replace("DÓNDE", "")
            total_text = total_text.replace("DONDE", "")
            total_text = total_text.replace("está", "")
            total_text = total_text.replace("ESTÁ", "")
            total_text = total_text.replace("ESTA", "")
            total_text = total_text.replace("messi", "")
            total_text = total_text.replace("Messi", "")
            total_text = total_text.replace("MESSI", "")
            total_text = total_text.replace("Donde", "")
            total_text = total_text.replace("donde", "")
            total_text = total_text.replace("esta", "")

        weirdPatterns = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u'\U00010000-\U0010ffff'
                                   u"\u200d"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\u3030"
                                   u"\ufe0f"
                                   u"\u2069"
                                   u"\u2066"
                                   u"\u200c"
                                   u"\u2068"
                                   u"\u2067"
                                   "]+", flags=re.UNICODE)
        total_text=  weirdPatterns.sub(r'', total_text)

        corpus_without_stopwords = ' '.join(
            [word for word in nltk.tokenize.word_tokenize(total_text) if word not in arabic_stop_words])
        corpus = arabic_reshaper.reshape(str(corpus_without_stopwords))
        corpus = get_display(corpus)
        wordcloud_dict = WordCloud().process_text(corpus)
        print(wordcloud_dict)
        # print(corpus)
        temp_dict = {}
        dict_list = []
        for k in wordcloud_dict:
            temp_dict = {}
            temp_dict["keyword"] = k
            temp_dict["value"] = wordcloud_dict[k]
            dict_list.append(temp_dict)

        df = pd.DataFrame.from_records(dict_list)
        df = df.sort_values("value", ascending=False)
        print(df.head(30))
        df.head(30).to_excel(r["_id"] + ".xlsx")
        result = apply_wordcloud(corpus, col_name,r["_id"], since, until)

def create_word_cloud_for_sentiment(col_name, since, until):
    col = db[col_name]
    result = col.aggregate([{"$match": {"created_at": {"$gte": since, "$lt": until},"sentiment": {"$nin": ["neutral", None]}}},{"$group": {"_id": "$sentiment", "count": {"$sum":1}}}, {"$sort": {"count": -1}}])

    arabic_stop_words = stopwords.words('english')
    arabic_stop_words.append("وهذا")
    arabic_stop_words.append("وفي")
    arabic_stop_words.append("من")
    arabic_stop_words.append("وانما")
    arabic_stop_words.append("الى")
    arabic_stop_words.append("بكل")
    arabic_stop_words.append("ان")
    arabic_stop_words.append("عندما")

    for r in result:
        print(r)
        total_text = ""
        emotion_result = list(col.find({"created_at": {"$gte": since, "$lt": until}, "sentiment": r["_id"]}))
        print(len(emotion_result))
        for d in emotion_result:
            total_text += d["t_text"] + " "

        total_text = re.sub(r"http\S+", "", total_text)
        total_text = total_text.replace("_", " ")
        total_text = total_text.replace("ميسي وينه", "")
        total_text = total_text.replace("السعودية", "")
        total_text = total_text.replace("السعوديه", "")
        total_text = total_text.replace("الارجنتين", "")
        total_text = total_text.replace("الأرجنتين", "")
        total_text = total_text.replace("Messi Winah", "")
        total_text = total_text.replace("Where Is Messi", "")
        total_text = total_text.replace("where is messi", "")
        total_text = total_text.replace("Where is messi", "")
        total_text = total_text.replace("Where is Messi", "")
        if col_name == "messi_es":
            total_text = total_text.replace("Dónde", "")
            total_text = total_text.replace("DÓNDE", "")
            total_text = total_text.replace("DONDE", "")
            total_text = total_text.replace("está", "")
            total_text = total_text.replace("ESTÁ", "")
            total_text = total_text.replace("ESTA", "")
            total_text = total_text.replace("messi", "")
            total_text = total_text.replace("Messi", "")
            total_text = total_text.replace("MESSI", "")
            total_text = total_text.replace("Donde", "")
            total_text = total_text.replace("donde", "")
            total_text = total_text.replace("esta", "")

        weirdPatterns = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u'\U00010000-\U0010ffff'
                                   u"\u200d"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\u3030"
                                   u"\ufe0f"
                                   u"\u2069"
                                   u"\u2066"
                                   u"\u200c"
                                   u"\u2068"
                                   u"\u2067"
                                   "]+", flags=re.UNICODE)
        total_text=  weirdPatterns.sub(r'', total_text)

        corpus_without_stopwords = ' '.join(
            [word for word in nltk.tokenize.word_tokenize(total_text) if word not in arabic_stop_words])
        corpus = arabic_reshaper.reshape(str(corpus_without_stopwords))
        corpus = get_display(corpus)
        wordcloud_dict = WordCloud().process_text(corpus)
        print(wordcloud_dict)
        # print(corpus)
        temp_dict = {}
        dict_list = []
        for k in wordcloud_dict:
            temp_dict = {}
            temp_dict["keyword"] = k
            temp_dict["value"] = wordcloud_dict[k]
            dict_list.append(temp_dict)

        df = pd.DataFrame.from_records(dict_list)
        df = df.sort_values("value", ascending=False)
        print(df.head(30))
        df.head(30).to_excel(r["_id"] + ".xlsx")
        result = apply_wordcloud(corpus, col_name,r["_id"], since, until)


def apply_wordcloud(corpus, col_name,file_path, since, until):
    background_image = np.array(Image.open('wordcloud3.jpg'))

    wordcloud = WordCloud(width=800, height=800,
                          mask = background_image,
                          background_color='white',
                          stopwords=stopwords.words('english'),
                          min_font_size=10, font_path='arial').generate(corpus)

    print(wordcloud)
    # plot the WordCloud image
    plt.figure(figsize=(10, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("AI_Analysis/" + col_name +"(" +file_path + ")_" + str(since)[:10] + "_" + str(until)[:10] + '.png')
    plt.show()
