import os
import time
import gc
import psutil
from tokenizer import RegexTokenizer
from tokenizer.logging import setup_logging, get_logger
from tokenizer.fast_text_loader import FastTextLoader

# Setup shared logging configuration (only done once globally)
setup_logging()

# Get logger for this module
logger = get_logger(__name__)

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def load_text_optimized(dataset: str, use_mmap: bool = True):
    """优化的文本加载函数"""
    loader = FastTextLoader()
    
    if dataset == "owt":
        filepath = "data/owt_train.txt"
        vocab_size = 32000
        model_name = "regex_32k"
        batch_size = 200000
        max_workers = 32
    elif dataset == "tiny":
        filepath = "data/TinyStoriesV2-GPT4-train.txt"
        vocab_size = 10000
        model_name = "regex_10k"
        batch_size = 200000
        max_workers = 32
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    file_size_gb = os.path.getsize(filepath) / (1024**3)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"File: {filepath} ({file_size_gb:.2f} GB)")
    logger.info(f"Memory before loading: {get_memory_usage():.1f} MB")
    
    t0 = time.time()
    
    # 选择加载方法
    if use_mmap:
        logger.info("Using memory mapping for fast loading...")
        text = loader.load_with_mmap(filepath)
    else:
        logger.info("Using chunked reading...")
        text = loader.load_chunked(filepath, chunk_size=16*1024*1024)  # 16MB chunks
    
    t1 = time.time()
    logger.info(f"Loaded text in {t1 - t0:.2f} seconds")
    logger.info(f"Text length: {len(text):,} characters")
    logger.info(f"Memory after loading: {get_memory_usage():.1f} MB")
    
    return text, vocab_size, model_name, batch_size, max_workers

def train_tokenizer_fast():
    """快速训练tokenizer"""
    # create a directory for models, so we don't pollute the current directory
    os.makedirs("models", exist_ok=True)
    
    dataset = "tiny"  # 或者 "tiny"
    
    try:
        # 加载文本
        text, vocab_size, model_name, batch_size, max_workers = load_text_optimized(
            dataset, use_mmap=True
        )
        
        # 构建tokenizer
        logger.info(f"Creating tokenizer with {max_workers} workers, batch size {batch_size}")
        tokenizer = RegexTokenizer(max_workers=max_workers, batch_size=batch_size)
        tokenizer.register_special_tokens({"<|endoftext|>": 100257})
        
        logger.info(f"Starting tokenizer training...")
        logger.info(f"Memory before training: {get_memory_usage():.1f} MB")
        
        t2 = time.time()
        tokenizer.train(text, vocab_size, verbose=True)
        t3 = time.time()
        
        logger.info(f"Trained tokenizer in {t3 - t2:.2f} seconds")
        logger.info(f"Memory after training: {get_memory_usage():.1f} MB")
        
        # 释放文本内存
        del text
        gc.collect()
        logger.info(f"Memory after cleanup: {get_memory_usage():.1f} MB")
        
        # 保存模型
        prefix = os.path.join("models", model_name)
        tokenizer.save(prefix)
        t4 = time.time()
        logger.info(f"Saved tokenizer in {t4 - t3:.2f} seconds")
        
        total_time = t4 - t2
        logger.info(f"Total training time: {total_time:.2f} seconds")
        
        return tokenizer
        
    except MemoryError:
        logger.error("内存不足！尝试使用更小的批次大小或流式处理")
        raise
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("开始快速tokenizer训练...")
        logger.info(f"Available CPU cores: {os.cpu_count()}")
        logger.info(f"Initial memory: {get_memory_usage():.1f} MB")
        
        tokenizer = train_tokenizer_fast()
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise 