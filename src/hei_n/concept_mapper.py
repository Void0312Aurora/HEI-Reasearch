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
            # Sememe: "AttributeValue|属性值" -> extract "attributevalue"
            # Concept: "C:apple:000000000123" -> extract "apple"
            
            if node.startswith('C:'):
                # Concept node: C:word:id
                parts = node.split(':')
                if len(parts) >= 2:
                    word = parts[1].lower()
                    self.word_to_id[word] = i
            elif '|' in node:
                # Sememe node: English|Chinese
                eng_part = node.split('|')[0].lower()
                self.word_to_id[eng_part] = i
            else:
                self.word_to_id[node.lower()] = i
                
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        print(f"Loaded {len(self.word_to_id)} concepts from checkpoint.")
        
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
        """Simple tokenization."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if w]
    
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
