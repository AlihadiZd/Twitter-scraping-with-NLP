from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import pymongo
import pandas as pd
import numpy as np
import re
from pathlib import Path
from nltk.corpus import stopwords

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['DB_name']


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def get_tfidf(data_col_name, langs, since, until):
    stop_words_list = []
    for l in langs:
        stop_words_list = stop_words_list + stopwords.words(l)

    corpus = []
    print(data_col_name)
    col = db[data_col_name]
    print(col.find({"created_at": {"$gte": since, "$lt": until}}).count())
    for d in col.find({"created_at": {"$gte": since, "$lt": until}}):
        text = d["full_text"]
        new = re.sub(r"http\S+", "", text)
        new = re.sub("@[A-Za-z0-9_]+", "", new)
        new = new.replace(' la ',' ')
        new = new.replace(' de ',' ')
        new = new.replace(' en ',' ')
        new = new.replace(' el ',' ')
        new = new.replace(' del ',' ')
        new = new.replace(' los ',' ')
        new = new.replace(' tu ',' ' )
        new = new.replace(' in ','' )
        new = new.replace(' to ',' ' )
        new = new.replace(' and ',' ' )
        new = new.replace(' في ',' ')

        new = new.replace('#','')
        new = new.replace('_', ' ')
        if new != '' and new != " ":
            corpus.append(new)

    db["tfidf_" + data_col_name +"_2"].delete_many({})

    for r in range(20):
        db["tfidf_" + data_col_name + "_2"].insert_one({"Rank": r + 1})

    for i in range(4):
        vectorizer = TfidfVectorizer(max_features=2000, min_df=2, max_df=0.7,stop_words = stop_words_list,ngram_range = (i+1,i+1) )
        Xtr = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        dense = Xtr.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        feature_array = np.array(feature_names)
        tfidf_sorting = np.argsort(Xtr.toarray()).flatten()[::-1]

        n = 15
        top_n = feature_array[tfidf_sorting][:n]
        result = top_mean_feats(Xtr, feature_names, grp_ids=None, min_tfidf=0.1, top_n=20)

        rank = 1
        for title, sheet in result.iterrows():
            print(sheet["feature"] + "         " + str(sheet["tfidf"]))
            db["tfidf_" + data_col_name+ "_2"].update_one({"Rank": rank}, {"$set": {str(i+1): sheet["feature"]}})
            rank+=1