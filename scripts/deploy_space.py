"""Upload the app to a Hugging Face Space (used by the GitHub Actions).

Needs HF_TOKEN (write token) and HF_SPACE ("username/space-name").
Training data stays in GitHub; the Space only gets what the app needs.
"""

import os
import sys

from huggingface_hub import HfApi

token, space = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE")
if not token or not space:
    print("::warning::Set the HF_TOKEN secret and HF_SPACE variable to deploy (see README).")
    sys.exit(0)

api = HfApi(token=token)
api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(
    repo_id=space, repo_type="space", folder_path=".",
    allow_patterns=["app.py", "README.md", "requirements.txt", "detector/*.py", "model/*"],
    delete_patterns=["detector/*.py", "model/*"],
    commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
)
print(f"Deployed to https://huggingface.co/spaces/{space}")
