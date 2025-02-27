from urllib.parse import urlparse
import re
import pandas as pd
import numpy as np


def extract_urls(text):
    if text != text:
        return np.nan
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    if len(urls) > 0:
        return urls
    else:
        return np.nan


def extract_domains(url_list):
    if url_list != url_list:
        return np.nan
    else:
        return [urlparse(url).netloc.replace("www.", "") for url in url_list if urlparse(url).scheme]

def parse_comments(posts_json, lim=False, debug=False):
    if lim:
        posts_json = posts_json[0:lim]
    IDs = [post.id for post in posts_json]
    texts = [post.text for post in posts_json]
    url_lists = [extract_urls(text) for text in texts]
    domain_lists = [extract_domains(url_list) for url_list in url_lists]

    posts = pd.DataFrame({
        "id":IDs,
        "text":texts,
        "url":url_lists,
        "domain":domain_lists,
    }) 
    
    posts["text"] = posts["text"].str.lower()
    posts["text"] = posts["text"].str.strip()
    posts = posts[posts["text"] != ""]
    if debug == True:
        posts["original_rank"] = [post.get("original_rank") for post in posts_json]

    return posts.reset_index(drop=True)
