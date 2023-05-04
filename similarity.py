import os
import json
import pymongo
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['Freelance_messi']
since = "2022-08-31"
until = "2022-11-01"

def get_text_similarities(text_array):
    clean_text_list = []
    for d in text_array:
        #list(docs_cursor)
        to_ignore = ["", " ", "  ", "\n"]
        new = d.replace("\n", " ")
        new = new.replace("   ", " ").replace("  ", " ")
        new = new.replace("جمعة مباركة", "")
        new = re.sub(r"http\S+", "", new)
        new = re.sub("@\S+", "", new)
        # new_ar = new.split(" ")
        # old_new = new
        # for w in new_ar:
        #     w = w.strip()
        #     if w in to_ignore:
        #         continue
        #     if w[0] == '#' or w[-1] == '#':
        #         new = new.replace(w + ' ', '')
        #     else:
        #         break
        #
        # for w in reversed(new_ar):
        #     w = w.strip()
        #     if w in to_ignore:
        #         continue
        #     if w[0] == '#' or w[-1] == '#':
        #         new = new.replace(' ' + w, '')
        #         new = new.replace('\n' + w, '')
        #     else:
        #         break
        #old_new = new
        #new = re.sub("#[،-٩_]+", "", new)
        # if new != old_new:
        #     print(old_new + "\n" + new)
        #     print("\n ------------------------------------- \n")
        #new = re.sub(r'\w*{}\w*'.format(hashtag), '', new)
        #new = re.sub(r"#(\w+)", "", new)
        if new not in to_ignore:
            clean_text_list.append(new)

    tfidf = TfidfVectorizer().fit_transform(clean_text_list)
    #pairwise_similarity = tfidf * tfidf.T
    ps = (tfidf * tfidf.T).toarray()
    after_formatting_count = len(ps)
    print(after_formatting_count)
    done_texts = []
    test_json = {"top_duplicated_tweets": []}
    docs_list= []
    for i in range(len(ps)):
        temp_json = {}
        if i in done_texts:
            continue

        temp_json["replicas"] = []
        for j in range(i, len(ps)):
            if ps[i][j] >= 0.6 and (j not in done_texts):
                temp_json["replicas"].append(clean_text_list[j])
                done_texts.append(j)
        done_texts.append(i)
        if len(temp_json["replicas"]) > 1:
            replicas_count = len(temp_json["replicas"])
            temp_json["count"] = replicas_count
            test_json["top_duplicated_tweets"].append(temp_json)
    #print('RAM memory5 % used:', psutil.virtual_memory()[2])
    test_json["top_duplicated_tweets"].sort(key=lambda x: x['count'], reverse=True)
    if len(test_json["top_duplicated_tweets"]) > 20:
        for sim in test_json["top_duplicated_tweets"][:20]:
            temp_json1 = {"text": sim["replicas"][0], "count": None}
            if len(sim["replicas"]) > 1:
                temp_json2 = {"text": sim["replicas"][1], "count": None}
                docs_list.append(temp_json2)
                if len(sim["replicas"]) > 2:
                    temp_json3 = {"text": sim["replicas"][2], "count": sim["count"]}
                    docs_list.append(temp_json3)
            temp_json4 = {"text": "\n\n", "count": None}
            docs_list.append(temp_json1)
            docs_list.append(temp_json4)
        results_df = pd.DataFrame(docs_list)
        print(results_df)
        #return test_json["top_duplicated_tweets"][:20]
        return results_df
    else:
        for sim in test_json["top_duplicated_tweets"]:
            temp_json1 = {"text": sim["replicas"][0], "count": None}
            if len(sim["replicas"])> 1:
                temp_json2 = {"text": sim["replicas"][1], "count": None}
                docs_list.append(temp_json2)
                if len(sim["replicas"]) > 2:
                    temp_json3 = {"text": sim["replicas"][2], "count": sim["count"]}
                    docs_list.append(temp_json3)
            temp_json4 = {"text": "\n\n", "count": None}
            docs_list.append(temp_json1)
            docs_list.append(temp_json4)
        results_df = pd.DataFrame(docs_list)
        print(results_df)
        #return test_json["top_duplicated_tweets"]
        return results_df

if __name__ == "__main__":
    col_name = ""
    docs_list = list(db[col_name].find({ "in_reply_to_status_id": None, "created_at": {"$gt": since, "$lt": until}}))
    text_list = []
    for d in docs_list:
        text_list.append(d["full_text"])
    df = get_text_similarities(text_list)
    df.to_excel("similarity_" + since[-5:] +"_" + until[-5:] +".xlsx")