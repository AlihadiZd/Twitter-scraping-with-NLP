'''Translating CNN dataset from english to arabic'''

from asyncio.log import logger
import time
import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from random import randint
import urllib.parse
import logging
import pymongo
import re

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['Freelance_messi']

xpath_google_trans = "/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div/div[8]/div/div[1]/span[1]"
lang_code = 'en '

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def translate_col(col_name, field_name, translated_field_name):
    col = db[col_name]
    options = webdriver.ChromeOptions()
    options.binary_location = "C:\Program Files\Google\Chrome\Application\chrome.exe"
    chrome_driver_binary = "chromedriver.exe"
    browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

    tweets_list = list(col.find({translated_field_name: {"$exists": False}}).skip(0).limit(2000))

    for d in tweets_list:
        sample_data_text_original= d[field_name]
        sample_data_text = re.sub(r"http\S+", "", sample_data_text_original)
        sample_data_text = re.sub("@[A-Za-z0-9_]+", "", sample_data_text)
        sample_data_text = re.sub("#(\w+)", "",sample_data_text)
        sample_data_text = re.sub("\n", " ",sample_data_text)
        if len(sample_data_text)>5000 or (not re.search('[a-zA-Zا-ى]', sample_data_text)):
            continue
        print(sample_data_text)
        sample_data_text= urllib.parse.quote_plus(sample_data_text)
        while True:
            try:
                browser.get("https://translate.google.co.in/?sl=auto&tl="+lang_code+"&text="+sample_data_text+"&op=translate")
                time.sleep(6)
                sample_output = browser.find_element("xpath", xpath_google_trans).text
                print(sample_output)
                col.update_one({"_id" : d["_id"]}, {"$set": {translated_field_name: sample_output}})
                print("-----------------------------------------")
                break
            except KeyboardInterrupt:
                exit()
            except BaseException as e:
                print(e)
                logger.error("Unknown error has occured, trying to retrieve data again...")
                try:
                    time.sleep(120)
                    browser.get(
                        "https://translate.google.co.in/?sl=auto&tl=" + lang_code + "&text=" + sample_data_text + "&op=translate")
                    sample_output = browser.find_element("xpath", xpath_google_trans).text
                    print(sample_output)
                    col.update_one({"_id": d["_id"]}, {"$set": {translated_field_name: sample_output}})
                    print("-----------------------------------------")
                    break
                except BaseException as e:
                    print(e)
                    time.sleep(1000)
    logging.info("Translation has finished.")

cols = ["messi_es", "messi_en", "messi_ar"]
for collection_name in cols:
    translate_col(collection_name, "full_text", "t_text")