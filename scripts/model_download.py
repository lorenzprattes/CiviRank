from huggingface_hub import snapshot_download
import os
from pathlib import Path

repos = [{"repo_id": "joaopn/glove-model-reduced-stopwords", "revision": None},
    {"repo_id": "PleIAs/celadon", "revision": "refs/pr/2"}]

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