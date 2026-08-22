"""诊断：加载 bge-reranker-v2-m3 的内存曲线与崩溃点。"""
import os
import time

os.environ.setdefault("TMP", os.path.join(os.getcwd(), ".tmp"))
os.environ.setdefault("TEMP", os.path.join(os.getcwd(), ".tmp"))

import psutil

proc = psutil.Process()


def mem(tag):
    free = psutil.virtual_memory().available / 1024 / 1024
    rss = proc.memory_info().rss / 1024 / 1024
    print(f"[{tag}] process RSS={rss:.0f}MB, system available={free:.0f}MB", flush=True)


mem("start")

print("step1: import torch...", flush=True)
import torch
print("torch threads:", torch.get_num_threads(), flush=True)
mem("after torch import")

print("step2: load tokenizer...", flush=True)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("models/bge-reranker-v2-m3")
mem("after tokenizer")

print("step3: load model fp32 low_cpu_mem_usage...", flush=True)
t0 = time.time()
model = AutoModelForSequenceClassification.from_pretrained(
    "models/bge-reranker-v2-m3", low_cpu_mem_usage=True
)
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
mem("after model load")

print("step4: inference smoke test...", flush=True)
model.eval()
with torch.no_grad():
    inputs = tok(
        [["测试问题", "退货运费由买家承担"]],
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    out = model(**inputs)
    print("logits:", out.logits, flush=True)
mem("after inference")
print("DIAGNOSTIC OK", flush=True)
