"""Download only the raw data files, not the code bundled with the datasets."""

from huggingface_hub import hf_hub_download

FILES = (
    ("PygmalionAI/PIPPA", "pippa_deduped.jsonl", "pippa"),
    ("Squish42/bluemoon-fandom-1-1-rp-cleaned", "data/full/bluemoon.train.json", "bluemoon"),
    ("Dampfinchen/Creative_Writing_Multiturn", "Creative-Writing-MultiturnV1.1.json", "creative"),
)

for repo, filename, folder in FILES:
    print(hf_hub_download(repo, filename, repo_type="dataset", local_dir=f"data/raw/{folder}"))
