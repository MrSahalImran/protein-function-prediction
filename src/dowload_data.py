import gdown
import os

URL = (
    "https://drive.google.com/file/d/12PQ08SJlLR1ZfBKJYVAgx6AZJiKxQnJp/view?usp=sharing"
)
OUTPUT = "data/swissprot_binary.csv"

os.makedirs("data", exist_ok=True)
gdown.download(URL, OUTPUT, quiet=False)

print("Dataset downloaded successfully.")
