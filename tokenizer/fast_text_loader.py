import os
import mmap
import time
from typing import Optional, Generator
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

class FastTextLoader:
    """快速加载大型文本文件的工具类"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
    
    def load_with_mmap(self, filepath: str, encoding: str = "utf-8") -> str:
        """使用内存映射加载文件 - 最快的方法"""
        print(f"Loading {filepath} with memory mapping...")
        start_time = time.time()
        
        with open(filepath, 'r', encoding=encoding) as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                text = mm.read().decode(encoding)
        
        end_time = time.time()
        print(f"Loaded {len(text):,} characters in {end_time - start_time:.2f} seconds")
        return text
    
    def load_chunked(self, filepath: str, chunk_size: int = 8192 * 1024, 
                    encoding: str = "utf-8") -> str:
        """分块读取文件，减少内存峰值"""
        print(f"Loading {filepath} in chunks of {chunk_size:,} bytes...")
        start_time = time.time()
        
        chunks = []
        with open(filepath, 'r', encoding=encoding, buffering=chunk_size) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        
        text = ''.join(chunks)
        end_time = time.time()
        print(f"Loaded {len(text):,} characters in {end_time - start_time:.2f} seconds")
        return text
    
    def load_parallel_chunks(self, filepath: str, num_workers: Optional[int] = None, 
                           encoding: str = "utf-8") -> str:
        """并行加载文件的不同部分"""
        if num_workers is None:
            num_workers = min(self.cpu_count, 8)  # 限制最大进程数
        
        print(f"Loading {filepath} with {num_workers} parallel workers...")
        start_time = time.time()
        
        # 获取文件大小
        file_size = os.path.getsize(filepath)
        chunk_size = file_size // num_workers
        
        def read_chunk(args):
            filepath, start, size, encoding = args
            with open(filepath, 'r', encoding=encoding) as f:
                f.seek(start)
                return f.read(size)
        
        # 创建任务
        tasks = []
        for i in range(num_workers):
            start = i * chunk_size
            size = chunk_size if i < num_workers - 1 else file_size - start
            tasks.append((filepath, start, size, encoding))
        
        # 并行执行
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunks = list(executor.map(read_chunk, tasks))
        
        text = ''.join(chunks)
        end_time = time.time()
        print(f"Loaded {len(text):,} characters in {end_time - start_time:.2f} seconds")
        return text
    
    def load_with_progress(self, filepath: str, encoding: str = "utf-8") -> str:
        """带进度显示的文件加载"""
        file_size = os.path.getsize(filepath)
        print(f"Loading {filepath} ({file_size / (1024**3):.2f} GB)...")
        
        start_time = time.time()
        chunks = []
        bytes_read = 0
        chunk_size = 8192 * 1024  # 8MB chunks
        
        with open(filepath, 'r', encoding=encoding, buffering=chunk_size) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                chunks.append(chunk)
                bytes_read += len(chunk.encode(encoding))
                
                # 显示进度
                progress = (bytes_read / file_size) * 100
                if len(chunks) % 100 == 0:  # 每100个块显示一次
                    print(f"Progress: {progress:.1f}% ({bytes_read / (1024**3):.2f} GB)")
        
        text = ''.join(chunks)
        end_time = time.time()
        print(f"Loaded {len(text):,} characters in {end_time - start_time:.2f} seconds")
        return text

# 使用示例和性能测试
def benchmark_loading_methods(filepath: str):
    """测试不同加载方法的性能"""
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        return
    
    loader = FastTextLoader()
    file_size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    print("=" * 50)
    
    methods = [
        ("Memory Mapping (推荐)", loader.load_with_mmap),
        ("Chunked Reading", loader.load_chunked),
        ("Parallel Chunks", loader.load_parallel_chunks),
        ("Progress Loading", loader.load_with_progress),
    ]
    
    for name, method in methods:
        print(f"\n测试方法: {name}")
        try:
            gc.collect()  # 清理内存
            text = method(filepath)
            print(f"成功加载 {len(text):,} 字符")
            del text  # 释放内存
            gc.collect()
        except Exception as e:
            print(f"失败: {e}")
        print("-" * 30)

if __name__ == "__main__":
    # 使用示例
    loader = FastTextLoader()
    
    # 方法1: 内存映射 (最快)
    # text = loader.load_with_mmap("data/owt_train.txt")
    
    # 方法2: 分块读取 (内存友好)
    # text = loader.load_chunked("data/owt_train.txt", chunk_size=16*1024*1024)
    
    # 方法3: 流式处理 (最节省内存)
    # for line in loader.stream_lines("data/owt_train.txt"):
    #     # 逐行处理文本
    #     pass
    
    # 性能测试
    benchmark_loading_methods("data/owt_train.txt") 