import os
os.environ['OMP_NUM_THREADS'] = '4'

import unsloth
from unsloth import FastLanguageModel

import json
import torch

from config import *
from data_utils import CLPsychDataMerger, Task12DataLoader
from rag_index import SequenceRAG

def truncate_to_n_words(text: str, n: int = SUMMARY_MAX_WORDS) -> str:
    words = text.strip().split()
    if len(words) <= n:
        return text
    return " ".join(words[:n])

def load_model_for_inference():
    print(f">>> Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    
    if os.path.exists(LORA_PATH):
        print(f">>> Loading LoRA adapter from: {LORA_PATH}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_PATH)
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(TEST_TASK3_PATH, "r", encoding="utf-8") as f:
        test_seqs = json.load(f)
    
    test12_loader = Task12DataLoader(
        TASK12_TEST_DIR,
        task1_pred_path=TASK1_TEST_PRED,
        task2_pred_path=TASK2_TEST_PRED
    )
    merger = CLPsychDataMerger(TEST_TASK3_PATH, test12_loader)
    
    rag = SequenceRAG(RAG_ENCODER)
    rag.load(RAG_INDEX_PATH)
    
    model, tokenizer = load_model_for_inference()
    
    predictions = []
    
    for seq in test_seqs:
        tid = seq["timeline_id"]
        sid = seq["sequence_id"]
        
        posts = merger.get_posts_for_sequence(seq)
        if not posts:
            print(f"[WARN] No posts found for {tid}/{sid}, skipping.")
            continue
        
        query_texts = [p.get("post", "") for p in sorted(posts, key=lambda x:x.get("post_index",0))]
        few_shots = rag.search([" | ".join(query_texts)], top_k=RAG_TOP_K)[0]
        
        prompt = merger.build_prompt(seq, posts, few_shots, change_type=None)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取 assistant 输出
        if "<|im_start|>assistant\n" in decoded:
            summary = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        else:
            summary = decoded.strip()
        
        summary = truncate_to_n_words(summary, SUMMARY_MAX_WORDS)
        
        predictions.append({
            "timeline_id": tid,
            "sequence_id": sid,
            "summary": summary
        })
        print(f"[Done] {sid}: {summary[:120]}...")
    
    with open(TASK3_PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f">>> Task 3.1 predictions saved to {TASK3_PRED_PATH}")

if __name__ == "__main__":
    main()