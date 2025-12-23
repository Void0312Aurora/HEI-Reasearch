"""
Build Semantic Edges for Phase II (Semantic Flow).
==================================================

Generates semantic edges (PMI and Definition) to augment the Cilin Skeleton.

Edge Types:
1. Skeleton (Base) - Already in checkpoint.
2. PMI (Context) - Wiki Co-occurrence (e.g., Doctor-Hospital).
3. Definition (Abstract) - Dictionary Definitions (e.g., Happy-Emotion).

Output:
    metrics/semantic_edges.pkl containing List[Tuple[u, v, weight, type]]
"""

import pickle
import os
import random
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.concept_mapper import ConceptMapper

def main():
    print("Building Semantic Edges for Phase II...")
    
    checkpoint_path = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    # Load Vocabulary from Checkpoint
    print(f"Loading vocabulary from {checkpoint_path}...")
    mapper = ConceptMapper(checkpoint_path=checkpoint_path)
    vocab = mapper.word_to_id
    id_to_word = mapper.id_to_word
    print(f"Vocab size: {len(vocab)}")
    
    semantic_edges = []
    
    # ---------------------------------------------------------
    # 1. Generate PMI Edges (Context / Co-occurrence)
    # ---------------------------------------------------------
    # Mocking high-quality PMI pairs for demonstration
    # In a real pipeline, this would consume a Wiki dump.
    
    pmi_pairs = [
        # Medical
        ("医生", "医院"), ("医生", "护士"), ("病人", "医院"), ("药", "吃"), ("感冒", "发烧"),
        # Education
        ("老师", "学生"), ("学校", "读书"), ("作业", "写"), ("考试", "复习"),
        # Weather
        ("下雨", "伞"), ("太阳", "热"), ("冬天", "冷"), ("雪", "白"),
        # Emotion & Action
        ("开心", "笑"), ("难过", "哭"), ("生气", "吵架"), ("害怕", "躲"),
        # Logic / Function
        ("如果", "那么"), ("因为", "所以"), ("虽然", "但是"),
        # Abstract
        ("梦想", "努力"), ("成功", "坚持"), ("失败", "放弃"),
        # Everyday
        ("饭", "吃"), ("水", "喝"), ("床", "睡"), ("车", "开")
    ]
    
    print(f"Generating PMI Edges (Mock: {len(pmi_pairs)} pairs)...")
    
    count_pmi = 0
    for w1, w2 in pmi_pairs:
        if w1 in vocab and w2 in vocab:
            u, v = vocab[w1], vocab[w2]
            # Type 2 = PMI, Weight = 1.0 (to be scaled by training loop)
            semantic_edges.append((u, v, 1.0, 2))
            # Bidirectional
            semantic_edges.append((v, u, 1.0, 2))
            count_pmi += 1
        else:
            # Try mapping synonyms?
            pass
            
    print(f"  Added {count_pmi} PMI edges (bidirectional total: {count_pmi*2}).")

    # ---------------------------------------------------------
    # 2. Generate Definition Edges (Abstract / Is-A / Description)
    # ---------------------------------------------------------
    # Linking concepts to their abstract definers or descriptors
    
    def_pairs = [
        # Emotions -> Types
        ("快乐", "情绪"), ("悲伤", "情绪"), ("愤怒", "情绪"),
        # Colors -> Visual
        ("红", "颜色"), ("蓝", "颜色"), ("白", "颜色"),
        # Shapes
        ("圆", "形状"), ("方", "形状"),
        # Animals -> Bio
        ("猫", "动物"), ("狗", "动物"), ("人", "动物"),
        # Roles
        ("老师", "职业"), ("医生", "职业"), ("学生", "身份")
    ]
    
    print(f"Generating Definition Edges (Mock: {len(def_pairs)} pairs)...")
    
    count_def = 0
    for w1, w2 in def_pairs:
        if w1 in vocab and w2 in vocab:
            u, v = vocab[w1], vocab[w2]
            # Type 3 = Definition, Weight = 1.0
            semantic_edges.append((u, v, 1.0, 3)) 
            # Definition is directed? u IS v. 
            # For graph embedding, we usually treat as undirected attraction, 
            # or directed if using specific loss. Assume undirected for now.
            semantic_edges.append((v, u, 1.0, 3))
            count_def += 1
            
    print(f"  Added {count_def} Definition edges.")
    
    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    output_path = "checkpoints/semantic_edges.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(semantic_edges, f)
        
    print(f"Saved {len(semantic_edges)} semantic edges to {output_path}.")

if __name__ == "__main__":
    main()
