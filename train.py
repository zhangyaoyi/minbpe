"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time


from minbpe import RegexTokenizer
from minbpe.regex import GPT4_SPLIT_PATTERN

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
# open some text and train a vocab of 512 tokens
#text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()
#text = open("data/owt_train.txt", "r", encoding="utf-8").read()
text = open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8").read()
t1 = time.time()
print(f"Loaded text in {t1 - t0:.2f} seconds")

# construct the Tokenizer object and kick off verbose training
tokenizer = RegexTokenizer(GPT4_SPLIT_PATTERN)
tokenizer.register_special_tokens({"<|endoftext|>": 100257})

print(f"Training tokenizer")
tokenizer.train(text, 10000, verbose=True)
t2 = time.time()
print(f"Trained tokenizer in {t2 - t1:.2f} seconds")
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "regex_10k")
tokenizer.save(prefix)
t3 = time.time()
print(f"Saved tokenizer in {t3 - t2:.2f} seconds")

print(f"Total time: {t3 - t0:.2f} seconds")