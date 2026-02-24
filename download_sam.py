from huggingface_hub import login, hf_hub_download
from dotenv import load_dotenv
import os

load_dotenv()
login(os.getenv("HF_TOKEN"))

path = hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    local_dir=".",                 # root directory
    local_dir_use_symlinks=False
)

print(path)