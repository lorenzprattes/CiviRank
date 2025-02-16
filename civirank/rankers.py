from civirank import analyzers, parsers
from ranking_challenge.request import RankingRequest
import numpy as np
import pandas as pd

class LocalRanker():
    def __init__(self, weights=None, lim=False, min_scores=0, debug=False, language="en"):

        # Set the weights for the different scores
        if weights is None:
            self.weights = {
                "no_toxicity": 1,
                "no_polarization": 1,
                "mtld": 0.5,
                "trustworthiness": 2,
                "prosociality": 1
            }
        else:
            self.weights = weights

        self.language=language
        self.TrustworthinessAnalyzer = analyzers.TrustworthinessAnalyzer()
        self.ToxicityAnalyzer = analyzers.ToxicityAnalyzer()
        self.ProsocialityPolarizationAnalyzer = analyzers.ProsocialityPolarizationAnalyzer(language=self.language)
        self.LexicalDensityAnalyzer = analyzers.LexicalDensityAnalyzer()

        # Scores that are considered in the compound score
        self.scores = ['no_toxicity', 'no_polarization', 'mtld', 'trustworthiness', 'prosociality']

        # Minimum number of scores a post needs to have to be considered in the compound score
        self.min_scores = min_scores

        # Limit the number of posts to be analyzed
        self.lim = lim

        # Debug flag
        self.debug = debug

    def rank_comments(self, comments, batch_size=16, scroll_warning_limit=-0.1):

        posts = parsers.parse_comments(comments, debug=self.debug)

        # Splits the posts into ones that get reranked and ones that don't
        parse_posts = posts[(posts.text.str.len() > 0)].copy()

        # Process posts
        parse_posts.loc[:, "trustworthiness"] = self.TrustworthinessAnalyzer.get_trustworthiness_scores(parse_posts)
        parse_posts.loc[:, "toxicity"] = self.ToxicityAnalyzer.get_toxicity_scores(parse_posts, batch_size=batch_size)
        parse_posts.loc[:, "polarization"] = self.ProsocialityPolarizationAnalyzer.get_similarity_polarization(parse_posts)
        parse_posts.loc[:, "prosociality"] = self.ProsocialityPolarizationAnalyzer.get_similarity_prosocial(parse_posts)
        parse_posts.loc[:, "mtld"] = self.LexicalDensityAnalyzer.get_mtld(parse_posts)
        for idx, row in parse_posts.iterrows():
            print(row["text"][:20]+"trust" +  str(row["trustworthiness"]) +"toxic" +  str(row["toxicity"]) +"polar" +  str(row["polarization"]) +"prosocia" +  str(row["prosociality"]) +"mltd" +  str(row["mtld"]), flush=True)
 
        parse_posts = analyzers.normalize(parse_posts)

        # Calculate the compound score
        parse_posts["compound_score"] = parse_posts[self.scores].apply(analyzers.calculate_compound_score, args=(self.weights, self.min_scores), axis=1)

        # Sort posts in descending order based on compound score
        ranked_posts = parse_posts.sort_values(by="compound_score", ascending=False)
        for idx, row in ranked_posts.iterrows():
            print(row["text"][:20]+"compound" + row["compound_score"] +"trust" +  row["trustworthiness"] +"toxic" +  row["no_toxicity"] +"polar" +  row["no_polarization"] +"prosocia" +  row["prosociality"], +"mltd" +  row["mtld"], flush=True)
        # extracts the first post with a compound score below the scroll_warning_limit
        insert_index = ranked_posts[ranked_posts['compound_score'] < scroll_warning_limit].first_valid_index()
        if insert_index is None:
            insert_index = -1

        ranked_dict = {row["id"]: idx for idx, row in ranked_posts.iterrows()}
        return ranked_dict, insert_index