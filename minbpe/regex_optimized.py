"""
Optimized Minimal (byte-level) Byte Pair Encoding tokenizer.
Multiple optimization techniques applied for better performance.
"""

import regex as re
from .base import Tokenizer, get_stats, merge
import time
from collections import Counter, defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os

# the main GPT text split patterns
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def process_chunk_batch(chunk_batch_data):
    """Process multiple chunks in a single worker to reduce overhead"""
    chunk_batch, word_freq_dict = chunk_batch_data
    stats = defaultdict(int)
    
    for chunk_ids, chunk in chunk_batch:
        chunk_weight = word_freq_dict.get(chunk, 1)
        # Inline get_stats for better performance
        for i in range(len(chunk_ids) - 1):
            pair = (chunk_ids[i], chunk_ids[i + 1])
            stats[pair] += chunk_weight
    
    return dict(stats)


def process_single_chunk_stats(chunk_data):
    """Original single chunk processing for backward compatibility"""
    chunk_ids, chunk, word_freq = chunk_data
    stats = {}
    get_stats(chunk_ids, stats, word_freq, chunk)
    return stats


class OptimizedRegexTokenizer(Tokenizer):
    
    def __init__(self, pattern=None, max_workers=None, batch_size=1000, 
                 min_freq_threshold=1, use_batching=True):
        """
        Optimized RegexTokenizer with multiple performance improvements
        
        Args:
            pattern: regex pattern for text splitting
            max_workers: number of worker processes (default: cpu_count)
            batch_size: number of chunks per batch for processing
            min_freq_threshold: minimum frequency for chunks to be processed
            use_batching: whether to use batch processing
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        
        # Optimization parameters
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Cap at 8 to avoid too much overhead
        self.batch_size = batch_size
        self.min_freq_threshold = min_freq_threshold
        self.use_batching = use_batching
        
        # Persistent process pool
        self.executor = None
        
    def __enter__(self):
        """Context manager support"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
    
    def _get_executor(self):
        """Lazy initialization of process pool"""
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self.executor
    
    def _process_stats_parallel_batched(self, ids, text_chunks, word_freq):
        """Optimized parallel processing with batching"""
        if self.use_batching and len(ids) > self.batch_size * 2:
            return self._process_with_batching(ids, text_chunks, word_freq)
        else:
            return self._process_without_batching(ids, text_chunks, word_freq)
    
    def _process_with_batching(self, ids, text_chunks, word_freq):
        """Process chunks in batches to reduce process overhead"""
        executor = self._get_executor()
        
        # Create batches
        batches = []
        word_freq_dict = dict(word_freq)  # Convert Counter to dict once
        
        for i in range(0, len(ids), self.batch_size):
            batch_end = min(i + self.batch_size, len(ids))
            chunk_batch = [(ids[j], text_chunks[j]) for j in range(i, batch_end)]
            batches.append((chunk_batch, word_freq_dict))
        
        # Submit all batches
        futures = [executor.submit(process_chunk_batch, batch_data) 
                  for batch_data in batches]
        
        # Collect results
        stats = defaultdict(int)
        for future in as_completed(futures):
            batch_stats = future.result()
            for pair, count in batch_stats.items():
                stats[pair] += count
        
        return dict(stats)
    
    def _process_without_batching(self, ids, text_chunks, word_freq):
        """Process chunks individually (for smaller datasets)"""
        executor = self._get_executor()
        
        # Prepare chunk data
        chunk_data = [(ids[j], text_chunks[j], word_freq.get(text_chunks[j], 1)) 
                     for j in range(len(ids))]
        
        # Process in parallel
        futures = [executor.submit(process_single_chunk_stats, data) 
                  for data in chunk_data]
        
        # Collect results
        stats = defaultdict(int)
        for future in as_completed(futures):
            chunk_stats = future.result()
            for pair, count in chunk_stats.items():
                stats[pair] += count
        
        return dict(stats)
    
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
            
            # Step 2: Count word frequency with Counter (more efficient)
            word_freq = Counter(text_chunks)
            t2 = time.time()
            if verbose:
                print(f"Total unique text chunks: {len(word_freq)}")
                print(f"Total text chunks before deduplication: {len(text_chunks)}")
                print(f"Time taken to count word frequency: {t2 - t1:.2f} seconds")
            
            # Step 3: Remove duplicates and apply frequency filtering
            text_chunks, word_freq = self._filter_low_frequency_chunks(
                list(word_freq.keys()), word_freq)
            
            if verbose:
                print(f"Text chunks after filtering (min_freq={self.min_freq_threshold}): {len(text_chunks)}")
                print(f"Time taken to filter: {time.time() - t2:.2f} seconds")
            
            # Step 4: Encode chunks
            ids = [list(ch.encode("utf-8")) for ch in text_chunks]
            t3 = time.time()
            if verbose:
                print(f"Time taken to encode text chunks: {t3 - t2:.2f} seconds")
                print(f"Starting training with {self.max_workers} workers, batch_size={self.batch_size}")
            
            # Step 5: Initialize process pool
            self._get_executor()
            
            # Step 6: Iterative merging with optimized parallel processing
            merges = {}
            vocab = {idx: bytes([idx]) for idx in range(256)}
            
            merge_times = []
            for i in range(num_merges):
                merge_start = time.time()
                
                # Parallel stats computation
                stats = self._process_stats_parallel_batched(ids, text_chunks, word_freq)
                
                if not stats:
                    if verbose:
                        print(f"No more pairs to merge at iteration {i+1}")
                    break
                
                # Find the pair with the highest count
                pair = max(stats.keys(), key=lambda k: stats[k])
                
                # Create new token
                idx = 256 + i
                
                # Replace all occurrences (this could be parallelized too)
                ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
                
                # Save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                
                merge_time = time.time() - merge_start
                merge_times.append(merge_time)
                
                # Progress reporting
                if verbose:
                    if i < 10 or (i + 1) % 100 == 0:
                        recent_times = merge_times[-100:] if len(merge_times) >= 100 else merge_times
                        avg_time = sum(recent_times) / len(recent_times)  # Last 100 merges average
                        eta = avg_time * (num_merges - i - 1)
                        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) "
                              f"had {stats[pair]} occurrences (avg_time: {avg_time:.3f}s, ETA: {eta:.1f}s)")
                    elif i == 10:
                        print("... (showing every 100th merge from now on)")
            
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
            # Cleanup resources
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
    
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
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
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


# Convenience alias
RegexTokenizer = OptimizedRegexTokenizer 