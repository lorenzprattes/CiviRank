language = "en"
model = ["celadon"]
 
from huggingface_hub import snapshot_download
import os
from pathlib import Path
import spacy

repos = [#{"repo_id": "joaopn/glove-model-reduced-stopwords", "revision": None},
    {"repo_id": "PleIAs/celadon", "revision": "refs/pr/2"}]

# Define repositories based on the language
model = None
if language == "en":
    model = "en_core_web_md"
elif language == "ger":
    model = "de_core_news_md"
else:
    raise ValueError(f"Unsupported language: {language}")
print(Path(__file__).resolve().parent)
print(Path(__file__).resolve().parent.parent)
#print ls of the current and parent dir
print(os.listdir(Path(__file__).resolve().parent))
print(os.listdir(Path(__file__).resolve().parent.parent))
parent_dir = Path(__file__).resolve().parent
filepath = parent_dir  / 'models'
try:
    spacy.cli.download(model)
except OSError:
    print("Model download failed")
# try:
#     nlp = spacy.load(filepath / model)
# except OSError:
#     spacy.cli.download(model)
#     nlp = spacy.load(model)
#     nlp.to_disk(filepath / model)



for repo in repos:
    if model not in repo["repo_id"]:
        continue
    try:
        model_folder = repo["repo_id"].replace("/", "_")
        model_path = filepath / model_folder
        if not os.path.exists(model_path):
            snapshot_download(repo_id=repo["repo_id"], local_dir=model_path, repo_type="model", revision=repo["revision"])
    except Exception as e:
      print("download failed")