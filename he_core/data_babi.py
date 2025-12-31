import torch
from torch.utils.data import Dataset, DataLoader
import os
import tarfile
import urllib.request
import re
from collections import Counter

# bAbI Dataset URL (v1.2)
URL = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"

class BabiDataset(Dataset):
    def __init__(self, data_root, task_id=1, train=True, download=True):
        self.data_root = data_root
        self.task_id = task_id
        self.train = train
        
        if download:
            self.download()
            
        self.stories, self.questions, self.answers = self.load_babi(data_root, task_id, train)
        
        # Build Vocab (handled externally or effectively here)
        # For simplicity, we'll build a static vocab on the fly if needed, 
        # but typically vocab should be shared between train/test.
        # Here we just store raw data, tokenization happens in collate or preprocessing.
        
    def download(self):
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        
        tar_path = os.path.join(self.data_root, "babi.tar.gz")
        if not os.path.exists(tar_path):
            print(f"Downloading bAbI dataset to {tar_path}...")
            urllib.request.urlretrieve(URL, tar_path)
            
        # Extract if needed
        extract_path = os.path.join(self.data_root, "tasks_1-20_v1-2")
        if not os.path.exists(extract_path):
            print("Extracting bAbI...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.data_root)
                
    def load_babi(self, data_root, task_id, train):
        type_str = "train" if train else "test"
        # Task ID format: qa1_..., qa2_...
        # We use en-10k (10k samples) for robustness
        path = os.path.join(data_root, "tasks_1-20_v1-2", "en-10k", f"qa{task_id}_*.txt")
        import glob
        files = glob.glob(path)
        target_file = [f for f in files if f"{type_str}.txt" in f][0]
        
        stories, questions, answers = [], [], []
        story = []
        
        with open(target_file, 'r') as f:
            for line in f:
                line = line.strip()
                nid, line = line.split(' ', 1)
                nid = int(nid)
                
                if nid == 1:
                    story = []
                    
                if '\t' in line: # Question
                    q, a, _ = line.split('\t')
                    # q also has '?' usually
                    substory = [x for x in story if x] # Copy current story
                    stories.append(" ".join(substory)) # Flatten story to single string
                    questions.append(q)
                    answers.append(a)
                else: # Story sentence
                    story.append(line)
                    
        return stories, questions, answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.stories[idx], self.questions[idx], self.answers[idx]

class BabiTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        
    def fit(self, texts):
        # texts: list of strings
        words = set()
        for t in texts:
            # Simple split, remove punctuation
            t = re.sub(r'([^\w\s])', r' \1 ', t) # separate punct
            for w in t.split():
                words.add(w.lower())
                
        for w in sorted(list(words)):
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                
    def encode(self, text, max_len=None):
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        indices = [self.word2idx.get(w.lower(), 1) for w in text.split()]
        if max_len:
            if len(indices) < max_len:
                indices += [0] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices):
        return [self.idx2word.get(idx.item(), "<unk>") for idx in indices]

def get_babi_loaders(task_id=1, batch_size=32, data_root='./data/babi', max_samples=None):
    train_ds = BabiDataset(data_root, task_id, train=True)
    test_ds = BabiDataset(data_root, task_id, train=False)
    
    if max_samples:
        train_ds.stories = train_ds.stories[:max_samples]
        train_ds.questions = train_ds.questions[:max_samples]
        train_ds.answers = train_ds.answers[:max_samples]
    
    # Build Vocab on Train + Test (to cover all answers)
    tokenizer = BabiTokenizer()
    all_text = train_ds.stories + train_ds.questions + train_ds.answers + \
               test_ds.stories + test_ds.questions + test_ds.answers
    tokenizer.fit(all_text)
    
    # Collate function
    def collate_fn(batch):
        stories, questions, answers = zip(*batch)
        
        # Max len in batch
        # We concatenate Story + Question for simplicity: "Story... Question?"
        inputs = [s + " " + q for s, q in zip(stories, questions)]
        max_len = max([len(re.sub(r'([^\w\s])', r' \1 ', t).split()) for t in inputs])
        
        # Pad to safe max or batch max
        # Let's cap at 50 for Task 1
        max_len = min(max_len, 100)
        
        # Encode inputs
        x = torch.stack([tokenizer.encode(t, max_len) for t in inputs])
        
        # Encode answers (treat as classification labels)
        # Babi tasks usually have single word answers
        y = torch.tensor([tokenizer.word2idx.get(a.lower(), 1) for a in answers], dtype=torch.long)
        
        return x, y
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader, tokenizer
