from tokenizer.base import Tokenizer
import multiprocessing as mp
import regex as re
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from collections import Counter, defaultdict

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    def __init__(self, max_workers = None, batch_size = 10000):
        super().__init__()
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.batch_size = batch_size
        self.executor = None

    def _get_executor(self):
        """Lazy initialization of process pool"""
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self.executor

    def _cleanup_executor(self):
        """Clean up the process pool"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

    def train(self, text, vocab_size, verbose=False):
        t0 = time.time()
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        try:
            # step 1: split text into chunks, for example, "hello world" -> ["hello", " world"]
            text_chunks = re.findall(self.compiled_pattern, text)
            t1 = time.time()
            if verbose:
                print(f"Total text chunks before deduplication: {len(text_chunks)}")
                print(f"Time taken to split text into chunks: {t1 - t0:.2f} seconds")

            # step 2: count word frequency, for example, ["hello", "world"] -> {"hello": 1, "world": 1}
            chunk_freq = Counter(text_chunks)
            t2 = time.time()
            if verbose:
                print(f"Time taken to count word frequency: {t2 - t1:.2f} seconds")

            # step 3: deduplicate chunks, for example, ["hello", "world", "hello"] -> ["hello", "world"]
            text_chunks = list(chunk_freq.keys())
            t3 = time.time()
            if verbose:
                print(f"Total unique text chunks after deduplication: {len(text_chunks)}")
                print(f"Time taken to deduplicate chunks: {t3 - t2:.2f} seconds")

            # step 4: encode chunks with UTF-8, for example, ["hello", " world"] -> [[104, 101, 108, 108, 111], [32, 119, 111, 114, 108, 100]]
            ids = [list(ch.encode("utf-8")) for ch in text_chunks]
            t4 = time.time()
            if verbose:
                print(f"Time taken to encode chunks: {t4 - t3:.2f} seconds")

            # step 5: initialize persistent process pool
            self._get_executor()

            # step 6: iteratively merge the most common pairs
            t_merge_start = time.time()
            merges = {}
            vocab = {idx: bytes([idx]) for idx in range(256)}
            
            for k in range(num_merges):
                executor = self._get_executor()

                #step 7: prepare the batched data
                chunk_batches = []
                for i in range(0, len(ids), self.batch_size):
                    batch_end = min(i + self.batch_size, len(ids))
                    chunk_batch = [(ids[j], chunk_freq[text_chunks[j]]) for j in range(i, batch_end)]
                    chunk_batches.append(chunk_batch)
                
                #step 8: calculate the subwords' frequency 
                futures = [executor.submit(process_chunk_batch, batch_data) 
                  for batch_data in chunk_batches]
                
                stats = defaultdict(int)
                for future in as_completed(futures):
                    batch_stats = future.result()
                    for pair, count in batch_stats.items():
                        stats[pair] += count
                
                #step 8: findout the pair with the highest count
                pair = max(stats.keys(), key=lambda n: stats[n])

                #step 9: replace the pair with the highest count with the new token
                idx = 256 + k
                ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

                # step 10: save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                
                if verbose:
                    print(f"merge {k+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
                    
            self.merges = merges
            self.vocab = vocab
        finally:
            self._cleanup_executor()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

def process_chunk_batch(batch_data):
    stats = defaultdict(int)
    for chunk_ids, chunk_freq in batch_data:
        for i in range(len(chunk_ids) - 1):
            pair = (chunk_ids[i], chunk_ids[i + 1])
            stats[pair] += chunk_freq    
    return stats

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids