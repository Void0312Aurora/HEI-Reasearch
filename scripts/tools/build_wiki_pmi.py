"""
Build Wiki PMI Edges (Real Scale) - Final Optimized Version (Hierarchical MapReduce).
=====================================================================================

Streams Wikipedia JSON, tokenizes using Cilin vocab, calculates PMI, 
and exports semantic edges for Interaction Engine Phase II.

Optimizations:
- Pool initializer to avoid repeated mapper serialization.
- Disk-based partial results to bypass multiprocessing IPC limits.
- **Hierarchical Reduce**: Parallel merge of partial results to maximize CPU usage during Reduce.
- **Resume Capability**: Can resume from post-reduce stage if temp files exist.
"""

import pickle
import os
import sys
import json
import math
import collections
import numpy as np
import time
import uuid
import shutil
from tqdm import tqdm
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

# Global mapper for each worker
_worker_mapper = None
_worker_window_size = 5
_worker_temp_dir = "checkpoints/temp_pmi"

def init_worker(checkpoint_path, window_size, temp_dir):
    """Initialize mapper once per worker process."""
    global _worker_mapper, _worker_window_size, _worker_temp_dir
    # Suppress output during worker init
    sys.stdout = open(os.devnull, 'w')
    
    from hei_n.concept_mapper import ConceptMapper
    if checkpoint_path: # Only needed for Map phase
        _worker_mapper = ConceptMapper(checkpoint_path=checkpoint_path)
    _worker_window_size = window_size
    _worker_temp_dir = temp_dir
    os.makedirs(_worker_temp_dir, exist_ok=True)
    
    sys.stdout = sys.__stdout__ # Restore

def process_chunk(lines):
    """
    Map Phase: Process a chunk of text lines.
    Saves results to disk and returns filename.
    """
    global _worker_mapper, _worker_window_size, _worker_temp_dir
    
    local_word_counts = collections.Counter()
    local_pair_counts = collections.defaultdict(int)
    processed = 0
    
    for line in lines:
        if not line: continue
        try:
            article = json.loads(line)
            text = article.get('text', '')
            if not text: continue
            
            target_ids = _worker_mapper.text_to_particles(text)
            if len(target_ids) < 2:
                continue
                
            processed += 1
            n = len(target_ids)
            for i in range(n):
                u_id = target_ids[i]
                local_word_counts[u_id] += 1
                start = max(0, i - _worker_window_size)
                end = min(n, i + _worker_window_size + 1)
                for j in range(start, end):
                    if i == j: continue
                    v_id = target_ids[j]
                    if u_id < v_id: pair = (u_id, v_id)
                    else: pair = (v_id, u_id)
                    local_pair_counts[pair] += 1
        except json.JSONDecodeError:
            continue
    
    if processed > 0:
        unique_id = uuid.uuid4().hex
        filename = os.path.join(_worker_temp_dir, f"part_{unique_id}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump((local_word_counts, local_pair_counts), f)
        return filename, processed
    else:
        return None, 0

def merge_pair(args):
    """
    Reduce Phase: Merge two partial files into one.
    """
    file_a, file_b = args
    if not file_a: return file_b
    if not file_b: return file_a
    
    # Load A
    with open(file_a, 'rb') as f:
        wc_a, pc_a = pickle.load(f)
        
    # Load B
    with open(file_b, 'rb') as f:
        wc_b, pc_b = pickle.load(f)
        
    # Merge into A
    wc_a.update(wc_b)
    for k, v in pc_b.items():
        pc_a[k] += v
        
    # Clean up B
    os.remove(file_b)
    
    # Overwrite A
    with open(file_a, 'wb') as f:
        pickle.dump((wc_a, pc_a), f)
        
    return file_a

def main():
    WIKI_FILE = "data/wiki/wikipedia-zh-20250901.json"
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    OUTPUT_FILE = "checkpoints/semantic_edges_wiki.pkl"
    TEMP_DIR = "checkpoints/temp_pmi"
    
    LIMIT_LINES = None
    WINDOW_SIZE = 5
    MIN_COUNT = 5
    TOP_K = 20
    CHUNK_SIZE = 5000  # Larger chunks reduce file count
    NUM_WORKERS = max(1, cpu_count() - 1)
    
    # Init
    print(f"[{time.strftime('%X')}] Script Launched.", flush=True)
    print(f"--- Build Wiki PMI (Hierarchical Parallel Reduce) ---", flush=True)
    
    # Check for RESUME capability
    resume_file = None
    if os.path.exists(TEMP_DIR):
        files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith('.pkl')]
        if len(files) == 1:
            resume_file = files[0]
            print(f">>> FOUND RECOVERY FILE: {resume_file}", flush=True)
            print(">>> Skipping Map/Reduce Phases and RESUMING...", flush=True)
    
    if not resume_file:
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    if not os.path.exists(WIKI_FILE) or not os.path.exists(CHECKPOINT):
        print("Error: Files not found.")
        return
        
    # Load Vocab Size
    print(f"[{time.strftime('%X')}] Loading Vocab Size...", flush=True)
    from hei_n.concept_mapper import ConceptMapper
    temp_mapper = ConceptMapper(checkpoint_path=CHECKPOINT)
    vocab_size = len(temp_mapper.word_to_id)
    del temp_mapper
    
    if not resume_file:
        # --- PHASE 1: MAP ---
        print(f"[{time.strftime('%X')}] Phase 1: Parallel Map ({NUM_WORKERS} workers, Chunk={CHUNK_SIZE})...", flush=True)
        
        partial_files = []
        total_processed = 0
        start_time = time.time()
        
        with open(WIKI_FILE, 'r', encoding='utf-8') as f:
            pool = Pool(NUM_WORKERS, initializer=init_worker, initargs=(CHECKPOINT, WINDOW_SIZE, TEMP_DIR))
            chunk = []
            futures = []
            
            # Batch reading loop
            for line in f:
                chunk.append(line)
                if len(chunk) >= CHUNK_SIZE:
                    futures.append(pool.apply_async(process_chunk, (chunk,)))
                    chunk = []
                    
                    # Flow Control
                    if len(futures) > NUM_WORKERS * 4:
                        active = []
                        for fut in futures:
                            if fut.ready():
                                fname, cnt = fut.get()
                                if fname: partial_files.append(fname)
                                total_processed += cnt
                            else:
                                active.append(fut)
                        futures = active
                        
                        if total_processed % 100000 == 0:
                             print(f"  > Mapped {total_processed:,} articles...", flush=True)
                
                if LIMIT_LINES and total_processed >= LIMIT_LINES: break
                
            if chunk: futures.append(pool.apply_async(process_chunk, (chunk,)))
            
            # Cleanup futures
            for fut in futures:
                fname, cnt = fut.get()
                if fname: partial_files.append(fname)
                total_processed += cnt
                
            pool.close()
            pool.join()
            
        print(f"[{time.strftime('%X')}] Map Complete. Generated {len(partial_files)} partial files.", flush=True)
        
        # --- PHASE 2: REDUCE (Treewise) ---
        print(f"[{time.strftime('%X')}] Phase 2: Parallel Tree Reduce...", flush=True)
        
        # Re-init pool without large Mapper
        pool = Pool(NUM_WORKERS, initializer=init_worker, initargs=(None, WINDOW_SIZE, TEMP_DIR))
        
        current_files = partial_files
        
        while len(current_files) > 1:
            print(f"  > Reducing level: {len(current_files)} files...", flush=True)
            pairs = []
            for i in range(0, len(current_files), 2):
                if i + 1 < len(current_files):
                    pairs.append((current_files[i], current_files[i+1]))
                else:
                    pairs.append((current_files[i], None))
                    
            next_level_files = pool.map(merge_pair, pairs)
            current_files = [f for f in next_level_files if f]
            
        pool.close()
        pool.join()
        
        final_file = current_files[0]
        print(f"[{time.strftime('%X')}] Reduce Complete. Loading final file...", flush=True)

    else:
        # Resume mode
        final_file = resume_file
        print(f"[{time.strftime('%X')}] Loaded partial data from {final_file}", flush=True)
    
    with open(final_file, 'rb') as f:
        global_word_counts, global_pair_counts = pickle.load(f)
        
    print(f"Total Unique Pairs: {len(global_pair_counts):,}", flush=True)
    
    # Stop Words (Functional/Structural Particles)
    STOP_WORDS = {
        "的", "是", "在", "和", "了", "有", "也", "被", "中", "对于", "等", 
        "为", "之", "与", "而", "以", "所", "但", "并", "将", "于", "对", 
        "就", "到", "自", "向", "由", "从", "或", "把", "让", "给", "得", 
        "着", "过", "去", "来", "说", "看", "想", "这", "那", "你", "我", 
        "他", "她", "它", "们", "个", "位", "只", "要", "能", "会", "应",
        "该", "可", "很", "更", "最", "好", "坏", "多", "少",
        "human", "person", "man", "woman", "thing", "stuff", "problem", "matter",
        "a", "an", "the", "is", "are", "of", "and", "or", "to", "in", "on", "at"
    }
    
    # --- PHASE 3: PMI ---
    print(f"[{time.strftime('%X')}] Calculating PMI (with Stopword Filtering)...", flush=True)
    total_tokens = sum(global_word_counts.values())
    log_total = math.log(total_tokens) if total_tokens > 0 else 0
    neighbors = collections.defaultdict(list)
    
    # Need Mapper to check stopwords (IDs -> Words)
    print("  > Reloading Mapper for Stopword Check...", flush=True)
    from hei_n.concept_mapper import ConceptMapper
    # Suppress output
    sys.stdout = open(os.devnull, 'w')
    final_mapper = ConceptMapper(checkpoint_path=CHECKPOINT)
    sys.stdout = sys.__stdout__
    id_to_word = final_mapper.id_to_word
    
    filtered_hubs = 0
    
    for (u, v), count in tqdm(global_pair_counts.items(), desc="PMI"):
        if count < MIN_COUNT: continue
        
        # STOPWORD CHECK
        if u in id_to_word:
            w_u = id_to_word[u]
            if w_u in STOP_WORDS:
                filtered_hubs += 1
                continue
        if v in id_to_word:
            w_v = id_to_word[v]
            if w_v in STOP_WORDS:
                filtered_hubs += 1
                continue
                
        count_u = global_word_counts[u]
        count_v = global_word_counts[v]
        pmi = math.log(count) - math.log(count_u) - math.log(count_v) + log_total
        if pmi > 0:
            neighbors[u].append((v, pmi))
            neighbors[v].append((u, pmi))
            
    print(f"  > Filtered {filtered_hubs} edges involving Stop Words.", flush=True)
            
    # --- PHASE 4: FILTER ---
    print(f"[{time.strftime('%X')}] Filtering Top-{TOP_K}...", flush=True)
    edges_to_save = []
    seen = set()
    
    # Corrected variable name usage
    active_nodes_set = set() 
    
    for u, nbrs in neighbors.items():
        nbrs.sort(key=lambda x: x[1], reverse=True)
        for v, s in nbrs[:TOP_K]:
            pair = (min(u,v), max(u,v))
            if pair in seen: continue
            
            # Decoupling: Save Strings instead of IDs
            if u in id_to_word and v in id_to_word:
                w_u = id_to_word[u]
                w_v = id_to_word[v]
                edges_to_save.append((w_u, w_v, s, 2))
                
                seen.add(pair)
                active_nodes_set.add(u)
                active_nodes_set.add(v)
            
    # Stats
    print("\n=== PMI Graph Health Check ===", flush=True)
    coverage = len(active_nodes_set) / vocab_size * 100.0 if vocab_size > 0 else 0
    print(f"Total Edges: {len(edges_to_save):,}", flush=True)
    print(f"Coverage: {coverage:.2f}%", flush=True)
    
    degree_map = collections.defaultdict(int)
    for u, v, s, t in edges_to_save:
        degree_map[u] += 1
        degree_map[v] += 1
    degree_vals = list(degree_map.values())
    
    if degree_vals:
        print(f"Degree: P50={np.percentile(degree_vals,50):.0f}, Max={np.max(degree_vals):.0f}", flush=True)
    print("==============================\n", flush=True)

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(edges_to_save, f)
    print(f"[{time.strftime('%X')}] Done. Saved to {OUTPUT_FILE}", flush=True)
    
    try: shutil.rmtree(TEMP_DIR)
    except: pass

if __name__ == "__main__":
    main()
