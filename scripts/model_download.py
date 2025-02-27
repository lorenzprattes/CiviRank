from huggingface_hub import snapshot_download
import os

repos = [{"repo_id": "joaopn/glove-model-reduced-stopwords", "revision": None},
    {"repo_id": "PleIAs/celadon", "revision": "refs/pr/2"}]

current_dir = os.path.dirname(__file__)

for repo in repos:
    try:
        model_folder = repo["repo_id"].replace("/", "_")
        filepath = os.path.join(current_dir, 'civirank', 'models', model_folder)
        if not os.path.exists(filepath):
            snapshot_download(repo_id=repo["repo_id"], local_dir=filepath, repo_type="model", revision=repo["revision"])
    except Exception as e:
        raise e

print("Downloaded models successfully")