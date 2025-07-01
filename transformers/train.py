from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import os

def train_bpe_tokenizer(corpus_files, vocab_size=30000, min_frequency=2):    


    model = models.BPE(unk_token="[UNK]")

    # 1. 初始化BPE模型
    tokenizer = Tokenizer(model)

    
    # 2. 设置预分词器（处理标点符号和空格）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)  # type: ignore
    
    # 3. 设置解码器
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore
    
    # 4. 设置后处理器（添加特殊token）
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)  # type: ignore
    
    # 5. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # 6. 训练分词器
    print(f"开始训练BPE分词器，词汇表大小: {vocab_size}")
    print(f"训练文件: {corpus_files}")
    
    tokenizer.train(corpus_files, trainer)
    
    print("BPE分词器训练完成！")
    return tokenizer

def save_tokenizer(tokenizer, save_directory):
    """保存分词器"""
    os.makedirs(save_directory, exist_ok=True)
    
    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_directory, "tokenizer.json"))
    
    # 转换为PreTrainedTokenizerFast并保存
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    
    fast_tokenizer.save_pretrained(save_directory)
    print(f"分词器已保存到: {save_directory}")

def load_tokenizer(tokenizer_path):
    """加载已训练的分词器"""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer

def test_tokenizer(tokenizer, test_texts):
    """测试分词器"""
    print("\n=== 分词器测试 ===")
    for text in test_texts:
        # 编码
        encoded = tokenizer.encode(text)
        tokens = tokenizer.tokenize(text)
        
        # 解码
        decoded = tokenizer.decode(encoded)
        
        print(f"原文: {text}")
        print(f"Token: {tokens}")
        print(f"Token IDs: {encoded}")
        print(f"解码: {decoded}")
        print("-" * 50)

# 主函数示例
if __name__ == "__main__":
    # 1. 创建示例语料（实际使用时替换为你的语料文件）
    corpus_file = "data/owt_train.txt"
    
    # 2. 训练BPE分词器
    tokenizer = train_bpe_tokenizer(
        corpus_files=[corpus_file],
        vocab_size=32000,  # 较小的词汇表用于演示
        min_frequency=2
    )
    
    # 3. 保存分词器
    save_directory = "bpe_tokenizer"
    save_tokenizer(tokenizer, save_directory)
    
    # 4. 加载并测试分词器
    loaded_tokenizer = load_tokenizer(save_directory)
    
    # 5. 测试分词效果
    test_texts = [
        "Hello, how are you doing today?",
        "This is a test sentence for tokenization.",
        "自然语言处理很有趣！",
        "BPE tokenizer works well for subword units."
    ]
    
    test_tokenizer(loaded_tokenizer, test_texts)
    
    # 6. 显示词汇表信息
    print(f"\n词汇表大小: {loaded_tokenizer.vocab_size}")
    print(f"特殊token: {loaded_tokenizer.special_tokens_map}")