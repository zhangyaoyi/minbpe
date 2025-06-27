# BPE 训练速度优化指南

## 🚀 核心问题分析

当前的并行化实现存在以下主要问题：

### 1. **重复创建进程池开销巨大**
```python
# 问题代码 (每次迭代都创建新进程池)
for i in range(num_merges):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = executor.map(self.process_chunk_stats, chunk_data)
```

**问题：**
- 每次合并迭代都要创建/销毁进程池
- 进程启动开销可能比实际计算时间还长
- 对于30k+次合并，这个开销是巨大的

### 2. **没有利用批处理**
- 每个进程只处理一个chunk
- 进程间通信开销过大
- 无法充分利用CPU cache

### 3. **没有过滤低频数据**
- 处理所有chunk，包括只出现1次的
- 浪费计算资源在对结果影响很小的数据上

## 💡 改进方案

### **方案1: 进程池复用 (立即可用)**

只需修改现有的并行化代码：

```python
class OptimizedRegexTokenizer(Tokenizer):
    def __init__(self, max_workers=None):
        super().__init__()
        # ... 其他初始化代码 ...
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.executor = None
    
    def _get_executor(self):
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self.executor
    
    def train(self, text, vocab_size, verbose=False):
        # ... 前面的代码 ...
        
        try:
            executor = self._get_executor()  # 复用同一个进程池
            
            for i in range(num_merges):
                chunk_data = [(chunk_ids, text_chunks[j], word_freq.get(text_chunks[j], 1)) 
                             for j, chunk_ids in enumerate(ids)]
                
                # 使用已存在的进程池
                results = executor.map(self.process_chunk_stats, chunk_data)
                
                # ... 合并逻辑 ...
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
```

**预期效果：** 2-3倍速度提升

### **方案2: 批处理优化 (中等难度)**

```python
def process_chunk_batch(chunk_batch_data):
    """一次处理多个chunk，减少进程间通信开销"""
    chunk_batch, word_freq_dict = chunk_batch_data
    stats = defaultdict(int)
    
    for chunk_ids, chunk in chunk_batch:
        chunk_weight = word_freq_dict.get(chunk, 1)
        # 内联get_stats逻辑以提高性能
        for i in range(len(chunk_ids) - 1):
            pair = (chunk_ids[i], chunk_ids[i + 1])
            stats[pair] += chunk_weight
    
    return dict(stats)

def _process_with_batching(self, ids, text_chunks, word_freq, batch_size=1000):
    executor = self._get_executor()
    
    # 创建批次
    batches = []
    word_freq_dict = dict(word_freq)
    
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        chunk_batch = [(ids[j], text_chunks[j]) for j in range(i, batch_end)]
        batches.append((chunk_batch, word_freq_dict))
    
    # 并行处理批次
    futures = [executor.submit(process_chunk_batch, batch_data) 
              for batch_data in batches]
    
    # 收集结果
    stats = defaultdict(int)
    for future in as_completed(futures):
        batch_stats = future.result()
        for pair, count in batch_stats.items():
            stats[pair] += count
    
    return dict(stats)
```

**预期效果：** 在方案1基础上再提升1.5-2倍

### **方案3: 频率过滤 (简单有效)**

```python
def _filter_low_frequency_chunks(self, text_chunks, word_freq, min_threshold=2):
    """过滤低频chunk，减少计算量"""
    filtered_chunks = [chunk for chunk in text_chunks 
                      if word_freq[chunk] >= min_threshold]
    filtered_freq = {chunk: word_freq[chunk] for chunk in filtered_chunks}
    return filtered_chunks, filtered_freq

# 在train方法中使用
text_chunks, word_freq = self._filter_low_frequency_chunks(
    list(word_freq.keys()), word_freq, min_threshold=2)
```

**预期效果：** 减少20-40%的计算量

### **方案4: 内存优化**

```python
# 使用更紧凑的数据结构
from collections import defaultdict

# 避免重复的字典查找
word_freq_dict = dict(word_freq)  # 转换一次，重复使用

# 使用生成器减少内存占用
def chunk_generator(ids, text_chunks, word_freq_dict):
    for j, chunk_ids in enumerate(ids):
        yield (chunk_ids, text_chunks[j], word_freq_dict.get(text_chunks[j], 1))
```

## 📊 性能对比预期

| 实现方式 | 相对原版速度 | 内存使用 | 实现难度 |
|----------|-------------|----------|----------|
| 原版串行 | 1x (基准) | 低 | - |
| 当前并行 | 1.5-2x | 中等 | - |
| 进程池复用 | 3-4x | 中等 | 简单 |
| + 批处理 | 5-8x | 中等 | 中等 |
| + 频率过滤 | 6-10x | 低 | 简单 |
| 全部优化 | 8-15x | 低 | 中等 |

## 🛠️ 实施建议

### **阶段1: 快速改进 (1小时内完成)**
1. 实现进程池复用
2. 添加频率过滤
3. 测试性能提升

### **阶段2: 深度优化 (半天)**
1. 实现批处理
2. 优化内存使用
3. 添加进度显示和ETA

### **阶段3: 高级优化 (可选)**
1. 考虑使用Cython/Numba加速核心函数
2. 实现增量统计更新
3. GPU加速（如果数据量很大）

## 📝 使用示例

```python
# 基础优化使用
from minbpe.regex_optimized import OptimizedRegexTokenizer

# 对于中等数据集
tokenizer = OptimizedRegexTokenizer(
    max_workers=4,
    batch_size=1000,
    min_freq_threshold=2
)

# 对于大数据集
tokenizer = OptimizedRegexTokenizer(
    max_workers=8,
    batch_size=2000,
    min_freq_threshold=5,
    use_batching=True
)

# 使用上下文管理器确保资源清理
with tokenizer:
    tokenizer.train(text, vocab_size=32000, verbose=True)
    tokenizer.save("model")
```

## ⚠️ 注意事项

1. **CPU核心数限制：** 不要设置超过8个worker，通常4-6个最优
2. **内存监控：** 大batch_size可能增加内存使用
3. **频率阈值：** min_freq_threshold=2通常是好的起点
4. **数据集大小：** 小数据集可能不需要所有优化

## 🔍 调试和监控

```python
# 添加详细的性能监控
def train_with_monitoring(self, text, vocab_size, verbose=True):
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Workers configured: {self.max_workers}")
    
    # ... 训练逻辑 ...
    
    if verbose and i % 100 == 0:
        mem_usage = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        print(f"Iteration {i}: Memory={mem_usage:.1f}MB, CPU={cpu_percent:.1f}%")
```

这些优化方案可以根据你的具体需求和数据集大小逐步实施。建议先从最简单的进程池复用开始！ 