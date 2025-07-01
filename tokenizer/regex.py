from tokenizer.base import Tokenizer, get_stats
import multiprocessing as mp
import regex as re
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from collections import Counter, defaultdict
from tokenizer.logging import get_logger
import gc

# Get logger for this module using shared configuration
logger = get_logger(__name__)

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Optimized pattern with better performance characteristics
OPTIMIZED_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]??\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    def __init__(self, max_workers = None, batch_size = 10000, use_optimized_pattern=True):
        super().__init__()
        # Choose pattern based on optimization flag
        pattern = OPTIMIZED_SPLIT_PATTERN if use_optimized_pattern else GPT4_SPLIT_PATTERN
        # Compile with performance flags
        self.compiled_pattern = re.compile(pattern, re.UNICODE | re.MULTILINE)
        # 移除线程数限制，使用全部CPU核心
        self.max_workers = max_workers or mp.cpu_count()
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

    def _split_text_parallel(self, text, chunk_size=100000):
        """Split large text in parallel chunks for faster processing"""
        if len(text) < chunk_size * 2:
            # For smaller texts, use single-threaded approach
            return re.findall(self.compiled_pattern, text)
        
        # Split text into chunks for parallel processing
        text_chunks = []
        chunk_start = 0
        
        while chunk_start < len(text):
            chunk_end = min(chunk_start + chunk_size, len(text))
            
            # Avoid splitting in the middle of words - find word boundary
            if chunk_end < len(text):
                # Look back for whitespace or punctuation
                while chunk_end > chunk_start and text[chunk_end] not in ' \t\n\r.,!?;:':
                    chunk_end -= 1
                if chunk_end == chunk_start:  # No boundary found, use original end
                    chunk_end = min(chunk_start + chunk_size, len(text))
            
            chunk = text[chunk_start:chunk_end]
            if chunk.strip():  # Only add non-empty chunks
                text_chunks.append(chunk)
            
            chunk_start = chunk_end
        
        # Process chunks in parallel
        executor = self._get_executor()
        futures = [executor.submit(_split_text_chunk, chunk, self.compiled_pattern) 
                  for chunk in text_chunks]
        
        # Collect results
        all_tokens = []
        for future in as_completed(futures):
            tokens = future.result()
            all_tokens.extend(tokens)
        
        return all_tokens

    def train(self, text, vocab_size, verbose=False):
        t0 = time.time()
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        try:
            # step 1: split text into chunks, for example, "hello world" -> ["hello", " world"]
            text_chunks = self._split_text_parallel(text)
            t1 = time.time()
            if verbose:
                logger.info(f"Total text chunks before deduplication: {len(text_chunks)}")
                logger.info(f"Time taken to split text into chunks: {t1 - t0:.2f} seconds")

            # step 2: count word frequency, for example, ["hello", "world"] -> {"hello": 1, "world": 1}
            chunks_freq = Counter(text_chunks)
            t2 = time.time()
            if verbose:
                logger.info(f"Time taken to count word frequency: {t2 - t1:.2f} seconds")

            # step 3: deduplicate chunks, for example, ["hello", "world", "hello"] -> ["hello", "world"]
            text_chunks = list(chunks_freq.keys())
            t3 = time.time()
            if verbose:
                logger.info(f"Total unique text chunks after deduplication: {len(text_chunks)}")
                logger.info(f"Time taken to deduplicate chunks: {t3 - t2:.2f} seconds")

            # step 4: 优化批处理，确保更均匀的负载分布
            chunks_batches = self._prepare_balanced_batches(text_chunks, chunks_freq, verbose)

            # release the memory
            del text_chunks
            del chunks_freq
            gc.collect()

            t4 = time.time()
            if verbose:
                logger.info(f"Time taken to batch chunks: {t4 - t3:.2f} seconds")

            # step 5: initialize persistent process pool
            self._get_executor()

            # step 6: iteratively merge the most common pairs
            merges = {}
            vocab = {idx: bytes([idx]) for idx in range(256)}
            
            for k in range(num_merges):
                executor = self._get_executor()

                #step 7: calculate the subwords' frequency 
                futures = [executor.submit(process_chunk_batch_optimized, batch_data) 
                  for batch_data in chunks_batches]
                
                stats = defaultdict(int)
                for future in as_completed(futures):
                    batch_stats = future.result()
                    for pair, count in batch_stats.items():
                        stats[pair] += count
                
                #step 8: findout the pair with the highest count
                if not stats:
                    break  # No more pairs to merge
                pair = max(stats.keys(), key=lambda n: stats[n])

                #step 9: 并行化merge操作
                idx = 256 + k
                # 将merge操作也并行化
                merge_futures = [executor.submit(apply_merge_to_batch, batch_data, pair, idx) 
                               for batch_data in chunks_batches]
                
                # 收集merge结果
                for i, future in enumerate(as_completed(merge_futures)):
                    batch_idx = merge_futures.index(future)
                    chunks_batches[batch_idx] = future.result()

                # step 10: save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                
                if verbose:
                    logger.info(f"merge {k+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
                    
            self.merges = merges
            self.vocab = vocab
        finally:
            self._cleanup_executor()

    def _prepare_balanced_batches(self, text_chunks, chunks_freq, verbose):
        """创建更均衡的批次，基于总字符数而不仅仅是chunk数量"""
        chunks_batches = []
        current_batch = []
        current_batch_size = 0
        target_batch_size = self.batch_size
        
        # 按频率排序，确保高频词汇分布均匀
        sorted_chunks = sorted(text_chunks, key=lambda x: chunks_freq[x], reverse=True)
        
        for chunk in sorted_chunks:
            chunk_id = list(chunk.encode("utf-8"))
            chunk_freq = chunks_freq[chunk]
            chunk_weight = len(chunk_id) * chunk_freq  # 基于实际工作量
            
            current_batch.append((chunk_id, chunk_freq, chunk))
            current_batch_size += chunk_weight
            
            # 当批次达到目标大小时，开始新批次
            if current_batch_size >= target_batch_size * 1000:  # 调整目标大小
                chunks_batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
                
                if verbose and len(chunks_batches) % 10 == 0:
                    logger.info(f"prepared batch {len(chunks_batches)} with balanced load")
        
        # 添加最后一个批次
        if current_batch:
            chunks_batches.append(current_batch)
        
        return chunks_batches

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
            ids = merge_optimized(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        if len(text) > 100000:  # Use parallel splitting for large texts
            text_chunks = self._split_text_parallel(text)
        else:
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

def process_chunk_batch_optimized(batch_data):
    """优化的批处理函数，使用更快的pair计数算法"""
    stats = defaultdict(int)
    for chunk_ids, chunk_freq, chunk in batch_data:
        # 使用更高效的循环避免重复索引访问
        prev_id = chunk_ids[0] if chunk_ids else None
        for curr_id in chunk_ids[1:]:
            pair = (prev_id, curr_id)
            stats[pair] += chunk_freq
            prev_id = curr_id
    return stats

def _split_text_chunk(text_chunk, compiled_pattern):
    """Helper function to split a text chunk using regex pattern"""
    return re.findall(compiled_pattern, text_chunk)

def merge_optimized(ids, pair, idx):
    """优化的merge函数，减少列表操作"""
    if not ids or len(ids) < 2:
        return ids
    
    newids = []
    i = 0
    pair0, pair1 = pair
    
    while i < len(ids):
        # 批量检查是否匹配，减少条件判断
        if i < len(ids) - 1 and ids[i] == pair0 and ids[i + 1] == pair1:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def apply_merge_to_batch(batch_data, pair, idx):
    """并行应用merge操作到整个批次"""
    merged_batch = []
    for chunk_ids, chunk_freq, chunk_text in batch_data:
        merged_ids = merge_optimized(chunk_ids, pair, idx)  # 使用优化版本
        merged_batch.append((merged_ids, chunk_freq, chunk_text))
    return merged_batch

def merge(ids, pair, idx):
    """
    原始merge函数，保持兼容性
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    return merge_optimized(ids, pair, idx)

def process_chunk_batch(batch_data):
    """原始批处理函数，保持兼容性"""
    return process_chunk_batch_optimized(batch_data)