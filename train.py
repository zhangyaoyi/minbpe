"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time


from minbpe import RegexTokenizer
from minbpe.regex import GPT2_SPLIT_PATTERN

# open some text and train a vocab of 512 tokens
#text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()
#text = open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8").read()
text = open("data/owt_train.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
# construct the Tokenizer object and kick off verbose training
tokenizer = RegexTokenizer(GPT2_SPLIT_PATTERN)

tokenizer.train(text, 10000, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "regex")
tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")