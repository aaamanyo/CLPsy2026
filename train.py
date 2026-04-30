import os
os.environ['OMP_NUM_THREADS'] = '4'

import unsloth
from unsloth import FastLanguageModel

import json
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from config import *
from data_utils import CLPsychDataMerger, Task12DataLoader
from rag_index import SequenceRAG

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/4] Loading training data...")
    task12_train = Task12DataLoader(TASK12_TRAIN_DIR)
    merger = CLPsychDataMerger(TASK3_TRAIN_PATH, task12_train)

    print("[2/4] Building RAG index...")
    task12_lookup = {}
    for (tid, pid), post in task12_train.posts.items():
        if tid not in task12_lookup:
            task12_lookup[tid] = {"posts": []}
        task12_lookup[tid]["posts"].append(post)

    rag = SequenceRAG(RAG_ENCODER)
    rag.build_from_task3(merger.task3, task12_lookup)
    rag.save(RAG_INDEX_PATH)

    print("[3/4] Building training prompts...")
    train_samples = []
    for idx, seq in enumerate(merger.task3):
        posts = merger.get_posts_for_sequence(seq)
        if not posts:
            continue
        few_shots = rag.search_excluding(seq, posts, exclude_idx=idx, top_k=RAG_TOP_K)
        prompt = merger.build_prompt(seq, posts, few_shots, change_type=seq.get("change_type"))
        completion = seq["summary"]
        full_text = prompt + completion + "<|im_end|>"
        if len(full_text) > MAX_SEQ_LENGTH * 4:
            continue
        train_samples.append({
            "text": full_text,
            "timeline_id": seq["timeline_id"],
            "sequence_id": seq["sequence_id"]
        })
    print(f"[Train] Total samples: {len(train_samples)}")
    dataset = Dataset.from_list(train_samples)

    print("[4/4] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_ratio=0.05,
            logging_steps=1,
            save_strategy="epoch",
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            seed=3407,
            report_to="none",
        ),
    )

    print(">>> Start training...")
    trainer.train()

    model.save_pretrained(LORA_PATH)
    tokenizer.save_pretrained(LORA_PATH)
    print(f">>> LoRA saved to {LORA_PATH}")

if __name__ == "__main__":
    main()