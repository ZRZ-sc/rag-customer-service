"""基准：bge-reranker-base 在 CPU 上的重排耗时（不同配置对比）。"""
import os
import time

import psutil
import torch

os.environ.setdefault("TMP", os.path.join(os.getcwd(), ".tmp"))
os.environ.setdefault("TEMP", os.path.join(os.getcwd(), ".tmp"))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "models/bge-reranker-base"
print(f"torch {torch.__version__}, threads={torch.get_num_threads()}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, low_cpu_mem_usage=True)
model.eval()
print(f"loaded, RSS={psutil.Process().memory_info().rss/1024/1024:.0f}MB", flush=True)

docs = [f"问题{i}：退货政策说明\n答案{i}：自签收次日起7天内可申请七天无理由退货，运费规则详见条款。" for i in range(50)]
query = "退货运费谁出？"

def bench(max_length, batch_size):
    pairs = [[query, d] for d in docs[:25]]
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            b = pairs[i:i+batch_size]
            inp = tok(b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            model(**inp)
    dt = time.time() - t0
    print(f"max_len={max_length} batch={batch_size} -> {dt:.2f}s for 25 pairs ({dt/25*1000:.0f}ms/pair)", flush=True)

bench(256, 32)
bench(512, 32)
bench(256, 8)

print("== int8 dynamic quantization ==", flush=True)
qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print(f"quantized RSS={psutil.Process().memory_info().rss/1024/1024:.0f}MB", flush=True)
with torch.no_grad():
    t0 = time.time()
    pairs = [[query, d] for d in docs[:25]]
    for i in range(0, len(pairs), 32):
        b = pairs[i:i+32]
        inp = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
        qmodel(**inp)
    dt = time.time() - t0
print(f"int8 max_len=256 batch=32 -> {dt:.2f}s for 25 pairs ({dt/25*1000:.0f}ms/pair)", flush=True)
print("BENCH DONE", flush=True)
