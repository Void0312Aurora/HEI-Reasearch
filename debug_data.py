import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from aurora.data.data_pipeline import CharStreamProcessor

def verify():
    print("Init Stream...")
    stream = CharStreamProcessor("data/CLUE/CLUECorpusSmall.txt")
    gen = stream.stream_batches(batch_size=10)
    batch = next(gen)
    print(f"Batch: {batch}")
    
    # Check stats
    from collections import Counter
    c = Counter(batch)
    print(f"Unique chars: {len(c)}")
    if len(c) < 5:
        print("WARNING: Low diversity!")
    else:
        print("SUCCESS: Data is diverse.")

if __name__ == "__main__":
    verify()
