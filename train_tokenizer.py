import os
import time

from tokenizer import RegexTokenizer
from tokenizer.logging import setup_logging, get_logger

# Setup shared logging configuration (only done once globally)
setup_logging()

# Get logger for this module
logger = get_logger(__name__)

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)
t0 = time.time()

dataset = "owt"

if dataset == "owt":
    text = open("data/owt_train.txt", "r", encoding="utf-8").read()
    vocab_size = 32000
    model_name = "regex_32k"
elif dataset == "tiny":
    text = open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8").read()
    vocab_size = 10000
    model_name = "regex_10k"

t1 = time.time()
logger.info(f"Loaded text in {t1 - t0:.2f} seconds")

# construct the Tokenizer object and kick off verbose training
tokenizer = RegexTokenizer(max_workers=16, batch_size=400000)
tokenizer.register_special_tokens({"<|endoftext|>": 100257})

logger.info(f"Training tokenizer")
tokenizer.train(text, vocab_size, verbose=True)
t2 = time.time()
logger.info(f"Trained tokenizer in {t2 - t1:.2f} seconds")
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", model_name)
tokenizer.save(prefix)
t3 = time.time()
logger.info(f"Saved tokenizer in {t3 - t2:.2f} seconds")

logger.info(f"Total time: {t3 - t0:.2f} seconds")