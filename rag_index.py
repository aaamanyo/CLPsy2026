import torch
import json
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
import os

class SequenceRAG:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = None
        self.sequences = []
        self.post_texts = []  # 缓存拼接文本
    
    def build_from_task3(self, task3_data: List[Dict], task12_lookup: Dict):
        """编码所有训练序列的帖子文本"""
        texts = []
        for seq in task3_data:
            tid = seq["timeline_id"]
            pids = seq.get("postids", [])
            # 从 task12_lookup 取帖子文本
            posts = task12_lookup.get(tid, {}).get("posts", [])
            pid_set = set(pids)
            seq_texts = [p["post"] for p in posts if p["post_id"] in pid_set]
            # 按原始 post_index 排序
            seq_texts = [p["post"] for p in sorted(posts, key=lambda x:x.get("post_index",0)) 
                         if p["post_id"] in pid_set]
            full = " | ".join(seq_texts)
            texts.append(full)
            self.sequences.append(seq)
        
        print(f"[RAG] Encoding {len(texts)} sequences with {self.model.get_sentence_embedding_dimension()}d...")
        self.embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        self.post_texts = texts
        print("[RAG] Index built.")
    
    def search(self, query_posts_texts: List[str], top_k: int = 2) -> List[List[Dict]]:
        """批量检索"""
        query_emb = self.model.encode(query_posts_texts, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(self.sequences)), dim=-1)
        
        results = []
        for indices in top_indices.cpu().numpy():
            results.append([self.sequences[int(i)] for i in indices])
        return results
    
    def search_excluding(self, query_seq: Dict, query_posts: List[Dict], 
                         exclude_idx: int, top_k: int = 1) -> List[Dict]:
        """训练时排除自身"""
        texts = [p.get("post", "") for p in sorted(query_posts, key=lambda x:x.get("post_index",0))]
        query_emb = self.model.encode(" | ".join(texts), convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        
        # 排除自身
        scores[exclude_idx] = -float("inf")
        top_k = min(top_k, len(self.sequences) - 1)
        if top_k <= 0:
            return []
        
        top_indices = torch.topk(scores, k=top_k).indices.cpu().numpy()
        return [self.sequences[int(i)] for i in top_indices]
    
    def save(self, path: str):
        torch.save({
            "embeddings": self.embeddings.cpu(),
            "sequences": self.sequences,
            "texts": self.post_texts
        }, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.embeddings = ckpt["embeddings"].cuda() if torch.cuda.is_available() else ckpt["embeddings"]
        self.sequences = ckpt["sequences"]
        self.post_texts = ckpt.get("texts", [])