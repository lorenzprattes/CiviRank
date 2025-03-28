import subprocess
import pandas as pd
from lexicalrichness import LexicalRichness
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os
from transformers import pipeline
from pathlib import Path
import spacy

class LexicalDensityAnalyzer():
    def __init__(self):
        pass

    def get_mtld(self, text):
         # copying approach from here https://github.com/notnews/unreadable_news
        assert type(text) in [str, pd.core.frame.DataFrame]
        if type(text) == str:
            lex = LexicalRichness(text)
            try:
                return lex.mtld()
            except ZeroDivisionError:
                return np.nan
        else:
            densities = []
            for i, row in text.iterrows():
                lex = LexicalRichness(row["text"])
                try:
                    densities.append(lex.mtld())
                except ZeroDivisionError:
                    densities.append(np.nan)
            return densities

class TrustworthinessAnalyzer():
    '''
        Class that loads a trustworthiness analyzer using the domain scores from https://doi.org/10.1093/pnasnexus/pgad286, avaiable at https://github.com/hauselin/domain-quality-ratings. It exposes a function to calculate the trustworthiness of links contained in a post. It returns a single floating point value between 0 and 1 as trustworthiness score. A higher value means a more trustworthy link. If multiple links are contained in a post and indexed in the NewsGuard data base, the average trustworthiness rating is returned. We remove the following domains from the csv file: youtube.com,facebook.com,google.com
    '''
    def __init__(self):
        fname = "domain_pc1.csv"
        parent_dir = Path(__file__).resolve().parent.parent
        filepath = parent_dir / 'data' / fname
        # print current working directory
        self.scores = pd.read_csv(filepath, usecols=["domain", "pc1"])
        self.scores = self.scores.set_index("domain")

    def extract_scores(self, domains):
        # no domains contained in text? 
        if domains != domains:
            return np.nan
        else:
            ratings = []
            for domain in domains:
                if domain in self.scores.index:
                    rating = self.scores.loc[domain]["pc1"]
                    ratings.append(rating)

            # domains contained in text but they are not news rated by NG?
            if len(ratings) == 0:
                return np.nan

            # domain(s) rated by NG contained in text: return average rating of
            # all rated domains
            else:
                return np.mean(ratings)

    def get_trustworthiness_scores(self, domains):
        assert type(domains) in [list, pd.core.frame.DataFrame]
        if type(domains) == list:
            return self.extract_scores(domains)
        else:
            scores = []
            for d in domains["domain"]:
                scores.append(self.extract_scores(d))
            return scores

class ToxicityAnalyzer():
    '''
        Class that loads a model to compute the toxicity of a text. It uses the unbiased toxic-roberta ONNX model from https://huggingface.co/protectai/unbiased-toxic-roberta-onnx. 
    '''
    def __init__(self, model_id = 'celadon'):
        self.device = torch.device("cpu")
        self.model_id = model_id
        if model_id == 'celadon':
            parent_dir = Path(__file__).resolve().parent.parent
            celadon_path = parent_dir / 'models' / "PleIAs_celadon"
            if not celadon_path.exists():
                raise FileNotFoundError(f"The specified path to celadon model '{celadon_path}' does not exist. Have you downloaded the model using model_download.py?")
            self.pipe = pipeline("text-classification", model=celadon_path, trust_remote_code=True)
        elif model_id == "detoxify":
            self.pipe = pipeline(
            'text-classification', 
            model='unitary/multilingual-toxic-xlm-roberta', 
            tokenizer='xlm-roberta-base', 
            function_to_apply='sigmoid', 
            return_all_scores=True
            )
        else:
            self.pipe = pipeline("text-classification", model=model_id)

    def get_toxicity_scores(self, texts):
        """ Analyze the given text or DataFrame and return toxicity scores """
        assert isinstance(texts, (str, pd.DataFrame)), "Input should be either a string or a DataFrame"
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts['text']:
            result = np.nan
            if self.model_id == 'celadon'or self.model_id == 'PleIAs/celadon':
                #this evaluation is based on the section "Pre-Training Data Curation" here https://arxiv.org/pdf/2410.22587
                scores = self.pipe(text)[0].values()
                total_score = sum(scores)
                max_score = max(scores)
                if total_score == 0:
                    result = 0
                elif total_score <= 3 and max_score <= 2:
                    result = total_score / 9
                elif (4 <= total_score <= 6): # Mild Toxicity
                    result = total_score / 9 
                elif (total_score == 3 and max_score == 3):# Mild Toxicity
                    result = 5 / 9
                elif total_score >= 7:   # Toxic Content
                    total_score = total_score / 9 if total_score < 9 else 1
                    result = 1
            if self.model_id == 'jagoldz/gahd':
                result = 1 if self.pipe(text)[0]['label'] == "LABEL_1" else 0
            if self.model_id == "textdetox/xlmr-large-toxicity-classifier":
                result = 1 if self.pipe(text)[0]['label'] == "toxic" else 0
            if self.model_id == "unitary/multilingual-toxic-xlm-roberta":
                result = self.pipe(text)[0]['toxic']
            results.append(result)

        return results

class ProsocialityPolarizationAnalyzer():
    '''
        Class that loads a model to compute the similarity of a text to a prosociality and a polarization dictionary. The similarity is computed as the cosine similarity between the text embeddings and the dictionary embeddings. 

        # Polarization
        Class that loads pre-calculated embeddings of the affective polarization dictionary from  https://academic.oup.com/pnasnexus/article/1/1/pgac019/6546199?login=false#381342977 and calculates similar embeddings using GloVe for a given text. It exposes a function get_similarity_polarization() that calculates the cosine similarity between the averaged dictionary embeddings and the text embedding following the DDR approach (see https://doi.org/10.3758/s13428-017-0875-9). The function returns a single floating point value between -1 and+1, with values closer to -1 meaning a text is less similar to polarizing language whereas values closer to +1 are more similar to polarizing language.

        # Prosociality
        Similar to the polarization class, it loads a dictionary of prosocial terms and calculates the cosine similarity between the averaged dictionary embeddings and the text embeddings. The function get_similarity_prosocial() returns a single floating point value between -1 and +1, with values closer to -1 meaning a text is less similar to prosocial language whereas values closer to +1 are more similar to prosocial language.
    '''

    def __init__(self, model_id = 'joaopn/glove-model-reduced-stopwords', label_filter = 'issue', language="en"):
        # Initialize the model
        parent_dir = Path(__file__).resolve().parent.parent
        filepath = parent_dir / 'models' / model_id.replace("/","_")
        if not filepath.exists:
            raise FileNotFoundError(f"The specified path to glove model '{filepath}' does not exist. Have you downloaded the model using model_download.py?")
        self.language = language
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        self.nlp = None
        if language == "en":
            self.nlp = spacy.load("en_core_web_md")
        if language == "ger":
            self.nlp = spacy.load("de_core_news_md")
        self.label_filter = label_filter
        # Load terms and compute their embeddings
        self.load_prosocial()
        self.load_polarization()


    def load_prosocial(self):
        # Load terms from CSV
        parent_dir = Path(__file__).resolve().parent.parent
        fname = "prosocial_dictionary_" + self.language + ".csv"
        filepath = parent_dir / 'data' / fname
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified path to prosocial dictionary '{filepath}' does not exist.")
        prosocial_dict = pd.read_csv(filepath, header=None, names = ['word'])
        prosocial_dict["word"] = prosocial_dict["word"].str.replace("*", "")
        prosocial_dict = list(prosocial_dict["word"].values)

        # Compute embeddings for the unique words
        self.dict_embeddings = self.model.encode(
            prosocial_dict,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # Average the embeddings to create a single dictionary embedding
        self.dict_embeddings_prosocial = torch.mean(self.dict_embeddings, dim=0)

    def load_polarization(self):
        # Load terms from CSV
        parent_dir = Path(__file__).resolve().parent.parent
        fname = "polarization_dictionary_" + self.language + ".csv"
        filepath = parent_dir /'data' / fname
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified path to polarization dictionary '{filepath}' does not exist.")
        df = pd.read_csv(filepath, header=0)
        if self.label_filter is not None:
            df = df[df['label'] == self.label_filter]
        unique_words = df['word'].unique()

        # Compute embeddings for the unique words
        self.dict_embeddings = self.model.encode(
            list(unique_words),
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # Average the embeddings to create a single dictionary embedding
        self.dict_embeddings_polarization = torch.mean(self.dict_embeddings, dim=0)

    def preprocess(self, df):
        # Regular expressions to clean up the text data
        df["text"] = df["text"].replace(
            to_replace=[r"(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})"],
            value=[""],
            regex=True,
        )
        df["text"] = df["text"].replace(to_replace=r"&.*;", value="", regex=True)
        df["text"] = df["text"].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True) 
        df["text"] = df["text"].replace(to_replace=r"\s+", value=" ", regex=True)
        df["text"] = df["text"].replace(to_replace=r"\@\w+", value="@user", regex=True)
        df["text"] = df["text"].apply(lambda x: " ".join([token.lemma_ for token in self.nlp(x)]))

    def get_embeddings(self, df):
        # Encode text in batches
        corpus_embeddings = self.model.encode(
            list(df["text"]),
            show_progress_bar=False,
            convert_to_tensor=True
        )
        assert len(corpus_embeddings) == len(df)
        return corpus_embeddings

    def get_similarity_prosocial(self, texts):
        df = texts.copy()
        self.preprocess(df)
        text_embeddings = self.get_embeddings(df)
        cos_sim = util.cos_sim(text_embeddings, self.dict_embeddings_prosocial)
        return cos_sim.cpu().numpy()

    def get_similarity_polarization(self, texts):
        df = texts.copy()
        self.preprocess(df)
        text_embeddings = self.get_embeddings(df)
        cos_sim = util.cos_sim(text_embeddings, self.dict_embeddings_polarization)
        return cos_sim.cpu().numpy()

def winsorize(val, bottom_limit, top_limit):
    if val < bottom_limit:
        return bottom_limit
    elif val > top_limit:
        return top_limit
    else:
        return val

def normalize(posts):
    '''
        We rescale all scores to be in the range [-1, 1] with negative values
        being undesirable and positive values being desirable. We also rename
        the reverted columns to avoid confusion.
    '''
    # scale polarization to be in [0, 1] for easier handling
    posts["polarization"] = (posts["polarization"] + 1) / 2
    # scale prosociality to be in [0, 1] for easier handling
    posts["prosociality"] = (posts["prosociality"] + 1) / 2
    # scale mtld to be in [0, 1] for easier handling
    posts["mtld"] = posts["mtld"] / posts["mtld"].max()

    # winsorize scores
    bottom_limit = 0.1
    top_limit = 0.9

    for col in ["polarization", "mtld", "prosociality"]:
        posts[col] = posts[col].apply(
            winsorize,
            args=(posts[col].quantile(q=bottom_limit),
                  posts[col].quantile(q=top_limit))
        )

        # rescale score to be in [0, 1] after removing outliers
        # this assumes that the score was in [0, 1] before winsorizing
        posts[col] = posts[col] - posts[col].min()
        posts[col] = posts[col] / posts[col].max()

    # revert score: high toxicity is good
    posts["toxicity"] = 1 - posts["toxicity"]
    # shift and rescale toxicity to be in [-1, 1]
    posts["toxicity"] = (posts["toxicity"] * 2) - 1

    # revert score: high polarization is good
    posts["polarization"] = 1 - posts["polarization"]
    # shift and rescale polarization to be in [-1, 1]
    posts["polarization"] = (posts["polarization"] * 2) - 1

    # shift and rescale prosociality to be in [-1, 1]
    posts["prosociality"] = (posts["prosociality"] * 2) - 1

    # shift and rescale mtld to be in [-1, 1]
    posts["mtld"] = (posts["mtld"] * 2) - 1

    # shift and rescale trustworthiness to be in [-1, 1]
    posts["trustworthiness"] = (posts["trustworthiness"] * 2) - 1

    posts = posts.rename(columns={"toxicity":"no_toxicity", "polarization":"no_polarization"})
    return posts

def calculate_compound_score(row, weights, min_scores, debug=False):
    if len(row.dropna()) < min_scores:
        return np.nan

    norm = 0
    compound_score = 0
    for score in weights.keys():
        if row[score] == row[score]: # nan-check
            compound_score += row[score] * weights[score]
            norm += weights[score]
    if norm != 0:
        return compound_score / norm
    else:
        return np.nan