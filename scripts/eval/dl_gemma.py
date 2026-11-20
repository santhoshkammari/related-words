import os, time
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("google/embeddinggemma-300m")
print("downloaded to", p, f"in {time.time() - t:.0f}s", flush=True)
print(sorted(os.listdir(p)))
