
import json
import jieba
import math
import sys
from collections import Counter
from tqdm import tqdm

# Hardcoded Chinese Stopwords (Aggressive)
STOPWORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就",
    "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "有", "个", "之",
    "为", "与", "及", "等", "或", "而", "但", "被",
    "让", "更", "做", "带", "给", "来", "把", "对",
    "从", "以", "向", "由", "它", "其", "此", "该",
    "\n", " ", "\u3000", "，", "。", "、", "；", "：",
    "“", "”", "（", "）", "《", "》", "！", "？",
    "-", "--", "...", ",", ".", "(", ")", "[", "]",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
])

def calculate_efficiency(file_path, limit_mb=100):
    print(f"Loading {file_path} (limit {limit_mb} MB)...")
    
    # Standard Tokenizer for Chinese
    def tokenize(text):
        # Jieba extraction in precise mode
        return [w for w in jieba.cut(text) if w not in STOPWORDS and len(w.strip()) > 0]

    # Stream processing
    total_bytes = 0
    limit_bytes = limit_mb * 1024 * 1024
    
    vocab = Counter()
    pair_counts = Counter()
    total_windows = 0
    window_size = 5
    
    with open(file_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=limit_bytes, unit='B', unit_scale=True)
        
        for line in f:
            line_bytes = len(line.encode('utf-8'))
            total_bytes += line_bytes
            pbar.update(line_bytes)
            
            try:
                data = json.loads(line)
                text = data.get("text", "")
            except:
                continue
                
            tokens = tokenize(text)
            
            # Update Vocab
            vocab.update(tokens)
            
            # Update Pairs (Sliding Window)
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i : i + window_size]
                unique_window = sorted(list(set(window))) # context set method
                
                # Pairwise interactions in window (Concept Co-occurrence)
                # We count undirected edges
                for idx1 in range(len(unique_window)):
                    w1 = unique_window[idx1]
                    for idx2 in range(idx1 + 1, len(unique_window)):
                        w2 = unique_window[idx2]
                        pair_counts[(w1, w2)] += 1
                        
                total_windows += 1
                
            if total_bytes >= limit_bytes:
                break
                
        pbar.close()

    print("Calculating PMI & Rigorous Metrics...")
    
    total_tokens = sum(vocab.values())
    total_events = sum(pair_counts.values())
    total_unique_pairs = len(pair_counts)
    
    if total_events == 0:
        return 0.0

    # 1. Apply c_min threshold
    c_min = 5
    filtered_pairs = {k: v for k, v in pair_counts.items() if v >= c_min}
    
    pmis = []
    high_pmi_unique = 0
    high_pmi_events = 0
    
    # PMI > 0.5 (approx 1.65 ratio)
    threshold_val = math.exp(0.5)
    
    print(f"Total Unique Pairs: {total_unique_pairs}")
    print(f"Pairs with count >= {c_min}: {len(filtered_pairs)}")
    
    for (w1, w2), count in filtered_pairs.items():
        freq1 = vocab[w1]
        freq2 = vocab[w2]
        
        # PMI Formula
        ratio = (count * total_tokens * total_tokens) / (freq1 * freq2 * total_events)
        
        # Avoid log(0)
        pmi = math.log(ratio + 1e-10)
        pmis.append(pmi)
        
        if score := ratio > threshold_val: # Check ratio directly for threshold
            high_pmi_unique += 1
            high_pmi_events += count
            
    # Metrics Calculation
    rho_unique = high_pmi_unique / total_unique_pairs if total_unique_pairs > 0 else 0.0
    rho_event = high_pmi_events / total_events 
    kappa = high_pmi_events / total_tokens if total_tokens > 0 else 0.0
    
    # Quantiles
    pmis.sort()
    if pmis:
        p50 = pmis[int(len(pmis)*0.5)]
        p90 = pmis[int(len(pmis)*0.9)]
        p99 = pmis[int(len(pmis)*0.99)]
    else:
        p50, p90, p99 = 0, 0, 0

    print(f"\n--- Gate A Rigorous Results (100MB Sample, c_min={c_min}) ---")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Events (Pairs): {total_events}")
    print(f"Unique Pairs: {total_unique_pairs}")
    print(f"\nMetrics:")
    print(f"  Rho (Unique): {rho_unique:.4f} (High PMI Unique / Total Unique)")
    print(f"  Rho (Event):  {rho_event:.4f}  (High PMI Events / Total Events)")
    print(f"  Kappa:        {kappa:.4f}   (Effective Events / Token)")
    print(f"\nPMI Distribution (for count >= {c_min}):")
    print(f"  P50: {p50:.2f}")
    print(f"  P90: {p90:.2f}")
    print(f"  P99: {p99:.2f}")
    
    # Pass Criteria (Revised)
    # If Rho(Event) > 0.10, we have enough signal.
    if rho_event > 0.10:
        print("PASS: Data Efficiency (Event-based) > 10%")
    else:
        print("FAIL: Data Efficiency too low.")
    
    return rho_event

if __name__ == "__main__":
    import sys
    # Default path
    path = "data/wiki_zh/wikipedia-zh-20250901.json"
    limit = 100
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    
    calculate_efficiency(path, limit_mb=limit)
