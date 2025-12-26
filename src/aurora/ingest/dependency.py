"""
Aurora Dependency Geometer.
===========================

Converts Natural Language text into Geometric Trajectories using Dependency Parsing.
Maps linguistic relations (nsubj, dobj) to Typed Transport Operators (U_subj, U_obj).

Note: Uses a Mock Parser for demonstration in environment without NLP libraries.
"""

import re
from typing import List, Dict, Tuple, Optional

class MockDependencyParser:
    """
    Simulates a Dependency Parser.
    Supports simple SVO patterns: "Subject Verb Object".
    """
    def parse(self, text: str) -> List[Dict]:
        """
        Returns list of tokens with head and dep relation.
        Format: [{'text': 'saw', 'id': 1, 'head': 0, 'dep': 'ROOT'}, ...]
        """
        tokens = text.strip().split()
        # Heuristic:
        # If 3 words: Subj Verb Obj
        # If 2 words: Subj Verb
        # If 4 words: Adj Subj Verb Obj?
        
        parsed = []
        for i, t in enumerate(tokens):
            parsed.append({'text': t, 'id': i+1, 'head': 0, 'dep': 'root'})
            
        n = len(tokens)
        if n == 3:
            # S V O
            # V is root (idx 2)
            # S (1) -> nsubj -> V (2)
            # O (3) -> dobj -> V (2)
            parsed[0]['head'] = 2; parsed[0]['dep'] = 'nsubj'
            parsed[1]['head'] = 0; parsed[1]['dep'] = 'root'
            parsed[2]['head'] = 2; parsed[2]['dep'] = 'dobj'
        elif n == 2:
            # S V
            parsed[0]['head'] = 2; parsed[0]['dep'] = 'nsubj'
            parsed[1]['head'] = 0; parsed[1]['dep'] = 'root'
        elif n >= 4:
            # Very dumb parser: Assume w[1] is Verb.
            # w[0] is Subj
            # w[2:] are Objects/Mods
            verb_idx = 1
            parsed[verb_idx]['head'] = 0; parsed[verb_idx]['dep'] = 'root'
            parsed[0]['head'] = verb_idx+1; parsed[0]['dep'] = 'nsubj'
            for i in range(2, n):
                 parsed[i]['head'] = verb_idx+1; parsed[i]['dep'] = 'dobj'
                 
        return parsed

class DependencyGeometer:
    def __init__(self):
        # Rel ID Mapping
        # 0: Neutral
        # 1: Subj->Verb (Inverse nsubj)
        # 2: Verb->Obj  (dobj)
        # 3: Mod->Head  (amod)
        self.REL_MAP = {
            'nsubj': 1,
            'dobj': 2,
            'amod': 3,
            'compound': 0,
            'root': 0
        }
        self.parser = MockDependencyParser()
        
    def text_to_edges(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Convert text to list of (u, v, rel_id) edges.
        Enforces Trajectory Direction:
        - Subject -> Verb (nsubj is inverted)
        - Verb -> Object (dobj is kept)
        - Modifier -> Head (amod is kept or inverted? "Red Apple". Red->Apple. Keep.)
        """
        parsed = self.parser.parse(text)
        edges = []
        
        # Build lookup
        idx_to_text = {t['id']: t['text'] for t in parsed}
        
        for token in parsed:
            if token['dep'] == 'root': continue
            
            head_id = token['head']
            dep_rel = token['dep']
            
            u_txt = token['text']
            v_txt = idx_to_text.get(head_id, "")
            
            if not v_txt: continue
            
            rel_id = self.REL_MAP.get(dep_rel, 0)
            
            # Geometric Direction Logic
            # Goal: Trajectory Flow.
            
            if dep_rel == 'nsubj':
                # Parse: Head(Verb) -> Dep(Subj)
                # Geom:  Subj -> Verb
                # So Edge: (Dep, Head)
                edges.append((u_txt, v_txt, rel_id))
                
            elif dep_rel == 'dobj':
                # Parse: Head(Verb) -> Dep(Obj)
                # Geom:  Verb -> Obj
                # So Edge: (Head, Dep)
                edges.append((v_txt, u_txt, rel_id))
                
            elif dep_rel == 'amod':
                # Parse: Head(Noun) -> Dep(Adj)
                # Geom:  Adj -> Noun (Red -> Apple)
                # So Edge: (Dep, Head)
                edges.append((u_txt, v_txt, rel_id))
                
            else:
                # Default: Head -> Dep? Or Dep -> Head?
                # Neural nets usually do Head->Dep.
                edges.append((v_txt, u_txt, 0))
                
        return edges

