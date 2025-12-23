"""
Concept Mapper for Aurora Interaction Engine.
=============================================

Maps natural language text to particle IDs in Aurora Base.
Uses OpenHowNet-derived vocabulary from training.
"""

import json
import re
from typing import List, Optional, Dict
import pickle

# Bilingual lexicon for English -> Chinese mapping
BILINGUAL_LEXICON = {
    # Food & Eating
    "food": "食物", "eat": "吃", "meal": "饭", "cook": "做饭", "rice": "米饭",
    "fruit": "水果", "vegetable": "蔬菜", "meat": "肉", "fish": "鱼",
    "apple": "苹果", "water": "水", "drink": "喝", "tea": "茶", "coffee": "咖啡",
    
    # Animals
    "animal": "动物", "dog": "狗", "cat": "猫", "bird": "鸟", "fish": "鱼",
    "horse": "马", "cow": "牛", "pig": "猪", "chicken": "鸡",
    
    # People & Relations
    "human": "人", "person": "人", "man": "男人", "woman": "女人", "child": "孩子",
    "friend": "朋友", "family": "家庭", "mother": "妈妈", "father": "爸爸",
    
    # Emotions
    "love": "爱", "happy": "快乐", "sad": "悲伤", "angry": "生气", "fear": "恐惧",
    "joy": "喜悦", "hate": "恨", "like": "喜欢", "dislike": "不喜欢",
    "开心": "快乐", "难过": "悲伤", "高兴": "快乐", "痛苦": "悲伤",
    
    # Nature
    "sun": "太阳", "moon": "月亮", "star": "星星", "sky": "天空", "earth": "地球",
    "mountain": "山", "river": "河", "sea": "海", "tree": "树", "flower": "花",
    
    # Actions
    "go": "去", "come": "来", "see": "看", "hear": "听", "say": "说",
    "think": "想", "know": "知道", "want": "想要", "need": "需要", "have": "有",
    
    # Intention Cluster (Gate M Fix)
    "desire": "想要", "require": "需要", "hope": "希望", "plan": "打算",
    "wish": "希望", "will": "想要", "would": "想要", "can": "能",
    "must": "必须", "should": "应该",
    
    # Time
    "time": "时间", "day": "天", "night": "夜", "today": "今天", "tomorrow": "明天",
    "year": "年", "month": "月", "week": "周",
    
    # Places
    "place": "地方", "home": "家", "school": "学校", "city": "城市", "country": "国家",
    
    # Common
    "good": "好", "bad": "坏", "big": "大", "small": "小", "new": "新", "old": "旧",
    "beautiful": "美丽", "ugly": "丑", "hot": "热", "cold": "冷",
    
    # Greetings
    "hello": "你好", "hi": "嗨", "goodbye": "再见", "thanks": "谢谢", "sorry": "对不起",
}


class ConceptMapper:
    """Map text to particle IDs and vice versa."""
    
    def __init__(self, vocab_path: str = None, checkpoint_path: str = None):
        """
        Initialize mapper with vocabulary.
        
        Args:
            vocab_path: Path to vocab.json (word -> particle_id)
            checkpoint_path: Path to Aurora checkpoint (to extract node names)
        """
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        
        if vocab_path:
            self._load_vocab(vocab_path)
        elif checkpoint_path:
            self._extract_vocab_from_checkpoint(checkpoint_path)
            
    def _load_vocab(self, path: str):
        """Load vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.word_to_id = json.load(f)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
    def _extract_vocab_from_checkpoint(self, checkpoint_path: str):
        """Extract vocabulary from checkpoint's node list."""
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            
        nodes = data.get('nodes', [])
        
        for i, node in enumerate(nodes):
            # Node format examples:
            # Sememe: "edible|食物" -> extract BOTH "edible" AND "食物"
            # Concept: "C:apple:000000000123" -> extract "apple"
            
            if node.startswith('C:'):
                # Concept node: C:word:id
                parts = node.split(':')
                if len(parts) >= 2:
                    word = parts[1].lower()
                    if word not in self.word_to_id:  # Don't overwrite
                        self.word_to_id[word] = i
            elif node.startswith('Code:'):
                # Cilin Code node: Code:Aa01
                code = node.split(':')[1]
                self.word_to_id[code.lower()] = i
            elif '|' in node:
                # Sememe node: English|Chinese (Legacy OpenHowNet)
                # Add BOTH parts to vocab
                parts = node.split('|')
                eng_part = parts[0].lower()
                zh_part = parts[1] if len(parts) > 1 else None
                
                if eng_part not in self.word_to_id:
                    self.word_to_id[eng_part] = i
                if zh_part and zh_part not in self.word_to_id:
                    self.word_to_id[zh_part] = i
            else:
                if node.lower() not in self.word_to_id:
                    self.word_to_id[node.lower()] = i
                
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        # print(f"Loaded {len(self.word_to_id)} concepts from checkpoint.")
        
    def save_vocab(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word_to_id, f, ensure_ascii=False, indent=2)
            
    def text_to_particles(self, text: str) -> List[int]:
        """
        Map text to particle IDs.
        
        Args:
            text: Input text (e.g., "apple fruit")
            
        Returns:
            List of particle IDs found in vocabulary.
        """
        # Simple tokenization (space-separated or character-based for Chinese)
        # For MVP, use simple word split
        words = self._tokenize(text)
        
        particle_ids = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.word_to_id:
                particle_ids.append(self.word_to_id[word_lower])
                
        return particle_ids
    
    def particles_to_text(self, ids: List[int]) -> List[str]:
        """
        Map particle IDs back to words.
        
        Args:
            ids: List of particle IDs
            
        Returns:
            List of corresponding words.
        """
        return [self.id_to_word.get(i, f"<UNK:{i}>") for i in ids]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenization with Forward Maximum Matching (FMM) for Chinese support.
        """
        # 1. Clean text
        text = re.sub(r'[^\w\s]', ' ', text).strip()
        
        # 2. Check if text contains Chinese (if so, use FMM)
        has_zh = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if not has_zh:
            # Simple English split
            words = text.split()
            translated = []
            for w in words:
                w_lower = w.lower()
                translated.append(BILINGUAL_LEXICON.get(w_lower, w))
            return translated
        else:
            # Chinese FMM
            # Remove spaces for FMM
            text_continuous = text.replace(" ", "")
            tokens = []
            max_len = 5 # Max word length to check
            
            i = 0
            while i < len(text_continuous):
                matched = False
                # Try longest match first
                for l in range(max_len, 0, -1):
                    if i + l > len(text_continuous):
                        continue
                    
                    sub = text_continuous[i : i+l]
                    # Check if sub is in vocab or lexicon keys (English keys, but we are parsing Chinese)
                    # Or lexicon values (Chinese mappings)
                    
                    # Check 1: In vocab directly
                    if sub in self.word_to_id:
                        tokens.append(sub)
                        i += l
                        matched = True
                        break
                        
                    # Check 2: In Lexicon Values (Reverse lookup? Too slow. We shouldn't need it if vocab is complete)
                    # But vocab keys are what we want.
                    
                if not matched:
                    # Input single char
                    tokens.append(text_continuous[i])
                    i += 1
            
            return tokens
    
    def find_similar(self, word: str, top_k: int = 5) -> List[str]:
        """Find similar words in vocabulary (fuzzy match)."""
        word_lower = word.lower()
        matches = []
        
        for vocab_word in self.word_to_id.keys():
            if word_lower in vocab_word or vocab_word in word_lower:
                matches.append(vocab_word)
                if len(matches) >= top_k:
                    break
                    
        return matches
