from pathlib import Path
import pandas as pd
import re
from civirank.utils import Comments, Comment
from math import sqrt
from googleapiclient import discovery
from openai import OpenAI
import json

class EvaluationDataGenerator():
  def __init__(self, path_to_folder):
    self.folder = Path.cwd() / path_to_folder

    self.path_to_comments_metadata = self.folder / "posts_metadata"
    if not self.path_to_comments_metadata.exists():
      raise FileNotFoundError(f"File {self.path_to_comments_metadata} not found")
    self.path_to_comments_texts = self.folder / "posts_texts"
    if not self.path_to_comments_texts.exists():
      raise FileNotFoundError(f"File {self.path_to_comments_texts} not found")
    self.path_to_articles = self.folder / "articles"
    if not self.path_to_articles.exists():
      raise FileNotFoundError(f"File {self.path_to_articles} not found")

    comments_metadata = pd.read_csv(self.path_to_comments_metadata, sep="\t")
    # clear of duplicates
    self.comments_metadata = comments_metadata.drop_duplicates()

    commments_texts = pd.read_csv(self.path_to_comments_texts, sep="\t", dtype=str)
    self.comments_texts = commments_texts.drop_duplicates()

  def get_comments_metadata(self):
    return self.comments_metadata.copy()

  def get_comments_texts(self):
    return self.comments_texts.copy()

  def find_article(self, id):
    with open(self.path_to_articles, "r") as f:
      for line in f:
        match = re.search(id, line)
        if match:
          url_match = re.search(r'http.*', line)
          if url_match:
            return url_match.group(0)

  def match_comments_of_article_to_metadata(self, id):
    comments_metadata_selection = self.comments_metadata[self.comments_metadata['Article_Id'] == id]
    comments_metadata_selection['Post_Id']
    comments_selected_with_metadata = pd.merge(
      comments_metadata_selection,
      self.comments_texts,
      how="inner",
      left_on="Post_Id",
      right_on="postid",
      validate="one_to_one"
    )
    return comments_selected_with_metadata

  def match_all(self):
    all = []
    for article_id in self.comments_metadata['Article_Id'].unique():
      url = self.find_article(article_id)
      matchedComments = self.match_comments_of_article_to_metadata(article_id)
      all.append({url: matchedComments})

  def extract_comments_with_metadata(self, article_id):
    comments_matched_to_metadata = self.match_comments_of_article_to_metadata(article_id)
    comments = []
    scores = []
    comments_matched_to_metadata = comments_matched_to_metadata.fillna("")
    for _, row in comments_matched_to_metadata.iterrows():
      comments.append(Comment(id=row["Post_Id"], text=row["text"]))
      scores.append({int(row["Votes_Pos"]), int(row["Votes_Neg"])})
    return Comments(comments=comments), comments_matched_to_metadata

  def extract_comments(self, article_id):
    commentsForArticle = self.match_comments_of_article_to_metadata(article_id)
    comments = []
    commentsForArticle = commentsForArticle.fillna("")
    for _, row in commentsForArticle.iterrows():
      comments.append(Comment(id=row["Post_Id"], text=row["text"]))
    return Comments(comments=comments)

  def calculate_score(self, comments_matched_to_metadata):
    comments_matched_to_metadata = comments_matched_to_metadata.copy()

    comments_matched_to_metadata["Votes_Pos"] = pd.to_numeric(comments_matched_to_metadata["Votes_Pos"], errors='coerce')
    comments_matched_to_metadata["Votes_Neg"] = pd.to_numeric(comments_matched_to_metadata["Votes_Neg"], errors='coerce')
    comments_matched_to_metadata["Score"] = comments_matched_to_metadata["Votes_Pos"] - comments_matched_to_metadata["Votes_Neg"]
    comments_matched_to_metadata.sort_values(by=['Score', 'Timestamp'], ascending=False, inplace=True)
    comments_matched_to_metadata = comments_matched_to_metadata.reset_index(drop=True)
    comments_matched_to_metadata["Score_Rank"]= comments_matched_to_metadata.index.copy()
    return comments_matched_to_metadata[["text", "Post_Id", "Score", "Score_Rank", "Timestamp", "Parent_Id", "Votes_Neg", "Votes_Pos"]]

  def calculate_reddit_score(self, comments_matched_to_metadata):
    comments_matched_to_metadata = comments_matched_to_metadata.copy()

    comments_matched_to_metadata["Votes_Pos"] = pd.to_numeric(comments_matched_to_metadata["Votes_Pos"], errors='coerce')
    comments_matched_to_metadata["Votes_Neg"] = pd.to_numeric(comments_matched_to_metadata["Votes_Neg"], errors='coerce')
    for _, row in comments_matched_to_metadata.iterrows():
      comments_matched_to_metadata.at[_, "Reddit_Score"] = self.confidence(row["Votes_Pos"], row["Votes_Neg"])
    comments_matched_to_metadata.sort_values(by=['Reddit_Score', 'Timestamp'], ascending=False, inplace=True)
    comments_matched_to_metadata = comments_matched_to_metadata.reset_index(drop=True)
    comments_matched_to_metadata["Reddit_Score_Rank"] = comments_matched_to_metadata.index.copy()
    return comments_matched_to_metadata[["Post_Id", "Reddit_Score", "Reddit_Score_Rank"]]

  #This code is taken from this blog post by Amir Salihefendic, describing reddits comment ranking algorithm back then.
  # https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
  def _confidence(self, ups, downs):
      n = ups + downs

      if n == 0:
          return 0

      z = 1.281551565545
      p = float(ups) / n

      left = p + 1/(2*n)*z*z
      right = z*sqrt(p*(1-p)/n + z*z/(4*n*n))
      under = 1+1/n*z*z

      return (left - right) / under

  def confidence(self, ups, downs):
      if ups + downs == 0:
          return 0
      else:
          return self._confidence(ups, downs)

  def calculate_all_scores(self, comments_matched_to_metadata, comments, civiranker):
    """
    Calculate and merge various scores for the given comments and metadata.

    Args:
      comments_matched_to_metadata (pd.DataFrame): DataFrame containing comments matched to their metadata.
      comments (object): An object containing comments data.
      civiranker (object): An instance of a CiviRanker class used to rank comments.

    Returns:
      pd.DataFrame: A DataFrame containing merged scores and warnings for each comment.
    """
    comments_matched_to_metadata = comments_matched_to_metadata.copy()
    score = self.calculate_score(comments_matched_to_metadata)
    reddit_score = self.calculate_reddit_score(comments_matched_to_metadata)
    ranked_dict, warning_index, ranked_posts = civiranker.rank(comments=comments.comments, debug=True)
    civirank = pd.DataFrame(
      [(post_id, dict["rank"], dict["score"]) for post_id, dict in ranked_dict.items()],
      columns=['Post_Id', 'CiviRank', 'CiviScore']
    ).sort_values(by='CiviRank', ascending=True)
    civirank["Warning"] = ["Warning" if rank > warning_index and warning_index != -1 else "No_Warning" for rank in civirank["CiviRank"]]

    merged = pd.merge(
      civirank,
      reddit_score,
      how="inner",
      left_on="Post_Id",
      right_on="Post_Id",
      validate="one_to_one"
    )
    merged = pd.merge(
      merged,
      score,
      how="inner",
      left_on="Post_Id",
      right_on="Post_Id",
      validate="one_to_one"
    )
    merged["Warning"] = merged["Warning"].astype(str)
    return merged

  @staticmethod
  def build_tree_sorted_by(scores, sort_by="Timestamp"):
    # Ensure scores is a DataFrame
    if not isinstance(scores, pd.DataFrame):
      scores = pd.DataFrame(scores)
      
    # note the very cool sorting by multiple columns in per column direction, as for timebased ranking we want most recent comments on top
    parents = scores[scores["Parent_Id"] == ""].sort_values(by=[sort_by, "Timestamp"], ascending=[True,False])
    non_parent_comments = scores[scores["Parent_Id"] != ""]
    tree_list = []
    def build_tree(tree_list, parents, non_parent_comments):
      for i, row in parents.iterrows():
        childs = non_parent_comments[non_parent_comments["Parent_Id"] == row["Post_Id"]]
        if not childs.empty:
          subtree = [row.to_dict()]
          if sort_by != "Timestamp":
            childs = childs.sort_values(by=[sort_by, "Timestamp"], ascending=[True, False])
          else:
            childs = childs.sort_values(by=sort_by, ascending=True)
          build_tree(subtree, childs, non_parent_comments)
          tree_list.extend(subtree)
        elif len(childs) == 0:
          tree_list.append(row.to_dict())
        non_parent_comments = non_parent_comments.drop(non_parent_comments[non_parent_comments["Parent_Id"] == row["Post_Id"]].index)

    build_tree(tree_list, parents, non_parent_comments)
    tree = pd.DataFrame(tree_list)
    return tree

  def extract_leaves_to_parent_scores(self, leaves, scores):
    tree = self.build_tree_sorted_by(scores, sort_by="CiviRank")
    tree["thread"] = -1 
    tree["childs"] = ""
    parents = tree[tree["Parent_Id"].eq("")]
    if len(parents) == len(tree):
      return
    for i, row in parents.iterrows():
      subtree = tree.loc[i+1:]
      if len(subtree) == 0:
        continue
      filtered_index = subtree[subtree["Parent_Id"].eq("")].index
      childs_over = filtered_index[0] if not filtered_index.empty else None
      if childs_over is None:
        if i == len(tree):
          continue
        elif i+ 1 == childs_over:
          continue
        else:
          childs_over = len(tree)
      elif childs_over - i <= 1:
        continue

      thread = tree.loc[i:childs_over-1]
      childs =  thread[1:]
      childs_avg = childs["CiviScore"].mean()
      childs_median = childs["CiviScore"].median()
      warning = childs["Warning"].eq("Warning").any()
      new_row = pd.DataFrame({
          "parent_id": [row["Post_Id"]],
          "parent_civiscore": [row["CiviScore"]],
          "childs_avg_civiscore": [childs_avg],
          "childs_median_civiscore": [childs_median],
          "warning": [warning],
          "childs_count": [len(childs)]
      })
      leaves = pd.concat([leaves, new_row], ignore_index=True)
    return leaves

class EvaluationCollector():
  def __init__(self, PERSPECTIVE_API_KEY, OPENAI_API_KEY):
    self.client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=PERSPECTIVE_API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )
    self.openai_client = OpenAI()
    self.openai_client.api_key = OPENAI_API_KEY
    self.prompt_template= """
          You are an advanced AI trained to evaluate text along five dimensions: toxicity, polarization, prosociality, trustworthiness, and informativeness.
          If a text is neutral regarding to a dimensions criterium, or no evaluation is possible, it is voted zero. If it decreases a dimension, it is negative. If it increases a dimension, it is positive.
          Therefore:
          A very toxic text would have a toxicity score close to 1.
          A very prosocial text would have a prosociality score close to 1.

          **Evaluation Criteria:**
          - **Toxicity (-1 to 1):** Measures the level of harmful, offensive, or aggressive language.
          - **Polarization (-1 to 1):** Measures how much the text divides opinions or incites conflict.
          - **Prosociality (-1 to 1):** Measures the level of positive, constructive, and community-oriented language.
          - **Trustworthiness (-1 to 1):** Measures the credibility and reliability of the information.
          - **Informativeness (-1 to 1):** Measures how well the text conveys useful and meaningful information.

          Please analyze the following text and return a structured JSON response with numerical ratings for each dimension.
          Respond **only** with a valid JSON object, without Markdown formatting or additional text.
          **Text to evaluate:**
          "{text}"

          **Output Format (JSON):**
          {{
              "toxicity": <score from -1 to 1>,
              "polarization": <score from -1 to 1>,
              "prosociality": <score from -1 to 1>,
              "trustworthiness": <score from -1 to 1>,
              "informativeness": <score from -1 to 1>
          }}
          """
  def get_prompt_template(self):
    return self.prompt_template

  def get_perspective_toxicity(self, text):
    analyze_request = {
      'comment': {'text': text},
      'requestedAttributes': {'TOXICITY': {}},
      'languages': ['de'],
      'doNotStore': True,
    }
    response = self.client.comments().analyze(body=analyze_request).execute()
    toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
    return toxicity_score

  def get_openai_moderation_scores(self, text):
    response = self.openai_client.moderations.create(
      model="omni-moderation-latest",
      input=text,
    )
    category_scores_dict = response.results[0].category_scores.model_dump()
    return category_scores_dict

  def get_openai_civi_scores(self, text):
    response = self.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an impartial evaluator that rates text based on predefined criteria."},
                  {"role": "user", "content": self.prompt_template.format(text=text)}],
        temperature=0  # Low temperature for consistent responses
    )
    response_dict = json.loads(response.choices[0].message.content)
    return response_dict

  def get_scores(self, text):
    #combine all dicts of above functions
    perspective_toxicity = self.get_perspective_toxicity(text)
    openai_moderation_scores = self.get_openai_moderation_scores(text)
    openai_civi_scores = self.get_openai_civi_scores(text)
    #combine all dicts of above functions into a single dict with one dim
    combined = {"perspective_toxicity": perspective_toxicity, **openai_moderation_scores, **openai_civi_scores}
    return combined