from huggingface_hub import snapshot_download
import os
from pathlib import Path
import argparse
import subprocess
import spacy

repos = [{"repo_id": "joaopn/glove-model-reduced-stopwords", "revision": None},
    {"repo_id": "PleIAs/celadon", "revision": "refs/pr/2"}]

# Define the argument parser
parser = argparse.ArgumentParser(description="Use this script to download the models for the Civirank project.")
parser.add_argument(
    "--language",
    default="en",
    help="Specify the language for the spacy pipeline (currently available: 'en', 'ger').",
    required=False
)

args = parser.parse_args()
language = args.language or os.getenv("LANGUAGE", "en")

# Define repositories based on the language
model = None
if language == "en":
    model = "en_core_web_md"
elif language == "ger":
    model = "de_core_news_md"
else:
    raise ValueError(f"Unsupported language: {args.language}")


parent_dir = Path(__file__).resolve().parent.parent
filepath = parent_dir / 'src' / 'civirank' / 'models'


for repo in repos:
    try:
        model_folder = repo["repo_id"].replace("/", "_")
        model_path = filepath / model_folder
        if not os.path.exists(model_path):
            snapshot_download(repo_id=repo["repo_id"], local_dir=model_path, repo_type="model", revision=repo["revision"])
    except Exception as e:
        raise e

print("Downloaded models successfully")


