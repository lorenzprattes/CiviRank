import spacy
from . import analyzers, parser
from .. import utils

class CiviRank():
    def __init__(self, weights=None, lim=False, min_scores=0, debug=False, language="en", model_id="celadon", scroll_warning_limit=None,):

        # Set the weights for the different scores
        if weights is None:
            self.weights = {
                "no_toxicity": 1,
                "no_polarization": 1,
                "mtld": 0.25,
                "trustworthiness": 2,
                "prosociality": 1
            }
        else:
            self.weights = weights

        self.language=language
        print(f"Language set to {language}", flush=True)
        self.nlp = None
        if language == "en":
            self.nlp = spacy.load("en_core_web_md")
        if language == "ger":
            self.nlp = spacy.load("de_core_news_md")



        #utils.download_models(language, model_id)
        self.TrustworthinessAnalyzer = analyzers.TrustworthinessAnalyzer()
        self.ToxicityAnalyzer = analyzers.ToxicityAnalyzer(model_id)
        print(f"ToxicityAnalyzer set to {model_id}", flush=True)
        self.ProsocialityPolarizationAnalyzer = analyzers.ProsocialityPolarizationAnalyzer(language=self.language)
        print("Prosociality and Polarization Analyzers initialized")
        self.LexicalDensityAnalyzer = analyzers.LexicalDensityAnalyzer()

        # Scores that are considered in the compounad score
        self.scores = ['no_toxicity', 'no_polarization', 'mtld', 'trustworthiness', 'prosociality']

        # Minimum number of scores a post needs to have to be considered in the compound score
        self.min_scores = min_scores

        self.scroll_warning_limit = 0

        if language == "en":
            self.scroll_warning_limit = -0.2
        elif language == "de":
            self.scroll_warning_limit = -0.2

        # Debug flag
        self.debug = debug
        print("Civirank initialized!", flush=True)

    def rank(self, comments, batch_size=16, scroll_warning_limit=None, debug=False):
        debug = debug or self.debug
        posts = parser.parse_comments(comments, debug=self.debug)
        if scroll_warning_limit is None:
            scroll_warning_limit = self.scroll_warning_limit
        # Splits the posts into ones that get reranked and ones that don't
        parse_posts = posts[(posts.text.str.len() > 0)].copy()

        # Process posts
        parse_posts["preprocessed_text"] = analyzers.preprocess(parse_posts["text"], nlp=self.nlp)

        parse_posts.loc[:, "trustworthiness"] = self.TrustworthinessAnalyzer.get_trustworthiness_scores(parse_posts)
        parse_posts.loc[:, "toxicity"] = self.ToxicityAnalyzer.get_toxicity_scores(parse_posts)
        parse_posts.loc[:, "polarization"] = self.ProsocialityPolarizationAnalyzer.get_similarity_polarization(parse_posts, col="preprocessed_text")
        parse_posts.loc[:, "prosociality"] = self.ProsocialityPolarizationAnalyzer.get_similarity_prosocial(parse_posts, col="preprocessed_text")
        parse_posts.loc[:, "mtld"] = self.LexicalDensityAnalyzer.get_mtld(parse_posts, col="preprocessed_text")

        parse_posts = analyzers.scale(parse_posts)

        # Calculate the compound score
        parse_posts["compound_score"] = parse_posts[self.scores].apply(analyzers.calculate_compound_score, args=(self.weights, self.min_scores, debug), axis=1)

        # Sort posts in descending order based on compound score
        ranked_posts = parse_posts.sort_values(by="compound_score", ascending=False)
        ranked_posts.reset_index(drop=True, inplace=True)

        # extracts the first post with a compound score below the scroll_warning_limit
        insert_index = ranked_posts[ranked_posts['compound_score'] < scroll_warning_limit].first_valid_index()
        if insert_index is None:
            insert_index = -1

        if debug:
            ranked_dict = {row["id"]: {"rank": idx, "score": row["compound_score"]} for idx, row in ranked_posts.iterrows()}
            for score in self.scores:
                ranked_posts[score] = ranked_posts[score].apply(lambda x: x * self.weights[score])

            return ranked_dict, insert_index, ranked_posts
        else:
            ranked_dict = {row["id"]: idx for idx, row in ranked_posts.iterrows()}
            return ranked_dict, insert_index

