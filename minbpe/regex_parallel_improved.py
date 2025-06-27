"""
Improved Parallel RegexTokenizer with process pool reuse.
This is a quick fix to the original parallel implementation.
"""

import regex as re
from .base import Tokenizer, get_stats, merge
import time
from collections import Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


# the main GPT text split patterns
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def process_chunk_stats(chunk_data):
    """Process a single chunk's statistics"""
    chunk_ids, chunk_weight = chunk_data
    stats = {}
    get_stats(chunk_ids, stats, chunk_weight)
    return stats


class ImprovedRegexTokenizer(Tokenizer):

    def __init__(self, pattern=None, max_workers=None, min_freq_threshold=1):
        """
        Improved RegexTokenizer with process pool reuse
        
        Args:
            pattern: regex pattern for text splitting
            max_workers: number of worker processes (default: min(cpu_count, 8))
            min_freq_threshold: minimum frequency for chunks to be processed
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        
        # Optimization parameters
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.min_freq_threshold = min_freq_threshold
        
        # Persistent process pool (key improvement!)
        self.executor = None

    def __enter__(self):
        """Context manager support"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        self._cleanup_executor()
    
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

    def _filter_low_frequency_chunks(self, text_chunks, word_freq):
        """Filter out low-frequency chunks to reduce computation"""
        if self.min_freq_threshold <= 1:
            return text_chunks, word_freq
            
        filtered_chunks = [chunk for chunk in text_chunks 
                          if word_freq[chunk] >= self.min_freq_threshold]
        filtered_freq = {chunk: word_freq[chunk] for chunk in filtered_chunks}
        
        return filtered_chunks, filtered_freq

    def train(self, text, vocab_size, verbose=False):
        t0 = time.time()
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        try:
            # Step 1: Split text into chunks
            text_chunks = re.findall(self.compiled_pattern, text)
            t1 = time.time()
            if verbose:
                print(f"Time taken to split text into chunks: {t1 - t0:.2f} seconds")

            # Step 2: Count word frequency
            word_freq = Counter(text_chunks)
            t2 = time.time()
            if verbose:
                print(f"Total unique text chunks: {len(word_freq)}")
                print(f"Total text chunks before deduplication: {len(text_chunks)}")
                print(f"Time taken to count word frequency: {t2 - t1:.2f} seconds")

            # Step 3: Apply frequency filtering and remove duplicates
            text_chunks, word_freq = self._filter_low_frequency_chunks(
                list(word_freq.keys()), word_freq)
            
            if verbose:
                print(f"Text chunks after filtering (min_freq={self.min_freq_threshold}): {len(text_chunks)}")
                print(f"Time taken to filter: {time.time() - t2:.2f} seconds")

            # Step 4: Encode text chunks
            ids = [list(ch.encode("utf-8")) for ch in text_chunks]
            t3 = time.time()
            if verbose:
                print(f"Time taken to encode text chunks: {t3 - t2:.2f} seconds")
                print(f"Starting parallel training with {self.max_workers} workers")

            # Step 5: Initialize persistent process pool (key improvement!)
            executor = self._get_executor()
            
            # Step 6: Iteratively merge the most common pairs
            merges = {}
            vocab = {idx: bytes([idx]) for idx in range(256)}
            
            merge_times = []
            for i in range(num_merges):
                merge_start = time.time()
                
                # Prepare chunk data for parallel processing
                chunk_data = [(ids[j], word_freq.get(text_chunks[j])) 
                             for j in range(len(ids))]
                
                # Use the persistent executor (no recreation overhead!)
                futures = [executor.submit(process_chunk_stats, data) for data in chunk_data]
                
                # Collect results
                stats = {}
                for future in as_completed(futures):
                    result = future.result()
                    for pair, count in result.items():
                        stats[pair] = stats.get(pair, 0) + count
                
                if not stats:
                    if verbose:
                        print(f"No more pairs to merge at iteration {i+1}")
                    break
                        
                # Find the pair with the highest count
                pair = max(stats.keys(), key=lambda k: stats[k])
                
                # Create new token
                idx = 256 + i
                
                # Replace all occurrences of pair in ids with idx
                ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
                
                # Save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                
                merge_time = time.time() - merge_start
                merge_times.append(merge_time)
                
                # Progress reporting with ETA
                if verbose:
                    if i < 10 or (i + 1) % 100 == 0:
                        recent_times = merge_times[-100:] if len(merge_times) >= 100 else merge_times
                        avg_time = sum(recent_times) / len(recent_times)
                        eta = avg_time * (num_merges - i - 1)
                        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) "
                              f"had {stats[pair]} occurrences (avg: {avg_time:.3f}s, ETA: {eta:.1f}s)")
                    elif i == 10:
                        print("... (showing every 100th merge)")

            # Save class variables
            self.merges = merges
            self.vocab = vocab
            
            if verbose:
                total_time = time.time() - t0
                print(f"Training completed in {total_time:.2f} seconds")
                if merge_times:
                    avg_merge_time = sum(merge_times) / len(merge_times)
                    print(f"Average time per merge: {avg_merge_time:.3f} seconds")
        
        finally:
            # Important: cleanup only at the end of training
            # This allows the process pool to be reused across all iterations
            pass  # We'll cleanup in __exit__ or when explicitly called

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
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
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        """
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
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids


# Convenience alias for backward compatibility
RegexTokenizer = ImprovedRegexTokenizer 