import os

TASK3_TRAIN_PATH = "train_task3.json"
TASK12_TRAIN_DIR = "task12_train_data"
TEST_TASK3_PATH  = "test_task3.json"
TASK12_TEST_DIR  = "task12_test_data"
TASK1_TEST_PRED  = "task1_pred.json"
TASK2_TEST_PRED  = "task2_pred.json"

OUTPUT_DIR       = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAG_INDEX_PATH   = os.path.join(OUTPUT_DIR, "rag_index.pt")
LORA_PATH        = os.path.join(OUTPUT_DIR, "qwen_psych_lora")
TASK3_PRED_PATH  = os.path.join(OUTPUT_DIR, "task3_pred.json")
TASK3B_PRED_PATH = os.path.join(OUTPUT_DIR, "TEAMNAME_Task3b.json")

BASE_MODEL  = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
RAG_ENCODER = "BAAI/bge-large-en-v1.5"

MAX_SEQ_LENGTH = 8192
LORA_R         = 128
LORA_ALPHA     = 64
LR             = 2e-4
EPOCHS         = 5
BATCH_SIZE     = 1
GRAD_ACCUM     = 4

RAG_TOP_K       = 2
MAX_NEW_TOKENS  = 700
TEMPERATURE     = 0.5
SUMMARY_MAX_WORDS = 350