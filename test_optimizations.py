#!/usr/bin/env python3
"""
Performance comparison script for different BPE tokenizer implementations.
"""

import time
import sys
import os
from pathlib import Path

# Add the minbpe directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "minbpe"))

def test_performance():
    """Test and compare different tokenizer implementations"""
    
    # Load test data
    try:
        # Try to load a smaller dataset for testing
        test_file = "data/TinyStoriesV2-GPT4-train.txt"
        if not os.path.exists(test_file):
            print(f"Warning: {test_file} not found. Creating sample text...")
            # Create sample text for testing
            sample_text = """
            This is a sample text for testing the BPE tokenizer optimizations.
            We want to see how different implementations perform with various text sizes.
            The quick brown fox jumps over the lazy dog. This sentence contains every letter.
            Machine learning and natural language processing are fascinating fields.
            Tokenization is a crucial step in text preprocessing for language models.
            """ * 1000  # Repeat to make it larger
        else:
            with open(test_file, 'r', encoding='utf-8') as f:
                sample_text = f.read()[:100000]  # First 100k characters for quick testing
        
        print(f"Testing with text length: {len(sample_text)} characters")
        print("=" * 60)
        
        # Test parameters
        vocab_size = 1000  # Smaller vocab for faster testing
        
        # Test 1: Original implementation
        print("1. Testing Original RegexTokenizer...")
        try:
            from regex import RegexTokenizer as OriginalTokenizer
            tokenizer1 = OriginalTokenizer()
            tokenizer1.register_special_tokens({"<|endoftext|>": vocab_size + 1})
            
            start_time = time.time()
            tokenizer1.train(sample_text, vocab_size, verbose=False)
            original_time = time.time() - start_time
            print(f"   Original implementation: {original_time:.2f} seconds")
        except ImportError as e:
            print(f"   Original implementation not available: {e}")
            original_time = float('inf')
        
        # Test 2: Parallel implementation
        print("\n2. Testing Parallel RegexTokenizer...")
        try:
            from regex_parallel import RegexTokenizer as ParallelTokenizer
            tokenizer2 = ParallelTokenizer()
            tokenizer2.register_special_tokens({"<|endoftext|>": vocab_size + 1})
            
            start_time = time.time()
            tokenizer2.train(sample_text, vocab_size, verbose=False)
            parallel_time = time.time() - start_time
            print(f"   Parallel implementation: {parallel_time:.2f} seconds")
        except ImportError as e:
            print(f"   Parallel implementation not available: {e}")
            parallel_time = float('inf')
        
        # Test 3: Optimized implementation
        print("\n3. Testing Optimized RegexTokenizer...")
        try:
            from regex_optimized import OptimizedRegexTokenizer
            
            # Test with different configurations
            configs = [
                {"max_workers": 4, "batch_size": 500, "min_freq_threshold": 1, "use_batching": True},
                {"max_workers": 4, "batch_size": 1000, "min_freq_threshold": 2, "use_batching": True},
                {"max_workers": 2, "batch_size": 2000, "min_freq_threshold": 1, "use_batching": False},
            ]
            
            optimized_times = []
            for i, config in enumerate(configs):
                print(f"   Config {i+1}: {config}")
                tokenizer3 = OptimizedRegexTokenizer(**config)
                tokenizer3.register_special_tokens({"<|endoftext|>": vocab_size + 1})
                
                start_time = time.time()
                tokenizer3.train(sample_text, vocab_size, verbose=False)
                opt_time = time.time() - start_time
                optimized_times.append(opt_time)
                print(f"   Optimized implementation (config {i+1}): {opt_time:.2f} seconds")
        
        except ImportError as e:
            print(f"   Optimized implementation not available: {e}")
            optimized_times = [float('inf')]
        
        # Summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY:")
        print("=" * 60)
        
        if original_time != float('inf'):
            print(f"Original implementation:     {original_time:.2f}s")
        
        if parallel_time != float('inf'):
            speedup = original_time / parallel_time if original_time != float('inf') else 1.0
            print(f"Parallel implementation:     {parallel_time:.2f}s (speedup: {speedup:.2f}x)")
        
        if optimized_times and optimized_times[0] != float('inf'):
            best_opt_time = min(optimized_times)
            speedup = original_time / best_opt_time if original_time != float('inf') else 1.0
            print(f"Best optimized config:       {best_opt_time:.2f}s (speedup: {speedup:.2f}x)")
        
        print("\nRecommendations:")
        if min(optimized_times) < parallel_time < original_time:
            print("✅ Use OptimizedRegexTokenizer for best performance")
            print("✅ Try different configurations based on your data size and hardware")
        elif parallel_time < original_time:
            print("✅ Use parallel implementation for decent speedup")
        else:
            print("ℹ️  For small datasets, original implementation might be sufficient")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_usage():
    """Demonstrate how to use the optimized tokenizer"""
    print("\n" + "=" * 60)
    print("USAGE DEMONSTRATION:")
    print("=" * 60)
    
    sample_text = "Hello world! This is a test of the optimized BPE tokenizer. " * 10
    
    try:
        from regex_optimized import OptimizedRegexTokenizer
        
        print("1. Basic Usage:")
        with OptimizedRegexTokenizer(max_workers=2, batch_size=100) as tokenizer:
            tokenizer.register_special_tokens({"<|endoftext|>": 1000})
            print("   Training tokenizer...")
            tokenizer.train(sample_text, vocab_size=300, verbose=True)
            
            # Test encoding/decoding
            test_text = "Hello world!"
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            print(f"   Original: {test_text}")
            print(f"   Encoded:  {encoded}")
            print(f"   Decoded:  {decoded}")
            
        print("\n2. Memory-efficient usage for large datasets:")
        print("""
        # For very large datasets, use:
        tokenizer = OptimizedRegexTokenizer(
            max_workers=8,           # Use more workers
            batch_size=2000,         # Larger batches
            min_freq_threshold=5,    # Filter rare chunks
            use_batching=True        # Enable batching
        )
        """)
        
    except ImportError:
        print("OptimizedRegexTokenizer not available")

if __name__ == "__main__":
    print("BPE Tokenizer Performance Testing")
    print("=" * 60)
    
    test_performance()
    demonstrate_usage()
    
    print("\n" + "=" * 60)
    print("Testing completed!") 