
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from aurora.data.data_pipeline import CharStreamProcessor

def test_continuity():
    print("=== Data Continuity Verification ===")
    # 1. Create a dummy file with known sequence
    dummy_text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    dummy_path = "tests/dummy_corpus.txt"
    with open(dummy_path, "w") as f:
        f.write(dummy_text)
        
    print(f"Created dummy corpus logic: {dummy_text[:10]}...")
    
    # 2. Initialize Stream
    # block_size small to force chunk reading
    stream = CharStreamProcessor(dummy_path, block_size=10)
    batch_gen = stream.stream_batches(batch_size=5)
    
    # 3. Consume Batches and verify strict order
    full_sequence_read = ""
    
    try:
        batch_idx = 0
        while True:
            batch = next(batch_gen)
            chunk_str = "".join(batch)
            print(f"Batch {batch_idx}: {chunk_str}")
            full_sequence_read += chunk_str
            batch_idx += 1
    except StopIteration:
        pass
        
    print(f"Original: {dummy_text}")
    print(f"Read In : {full_sequence_read}")
    
    if dummy_text == full_sequence_read:
        print("SUCCESS: Data pipeline is strictly continuous.")
        return True
    else:
        print("FAILURE: Data pipeline reordered or lost data.")
        return False

if __name__ == "__main__":
    if test_continuity():
        sys.exit(0)
    else:
        sys.exit(1)
