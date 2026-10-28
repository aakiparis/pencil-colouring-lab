"""Shared immutable identity for eight-GPU calibration, pilot and recovery."""
import hashlib,json
from pathlib import Path
RUN_FILES=['pilot.json','frozen-dataset/manifest.json','vendor/train_dreambooth_lora_qwenimage21_img2img.py','trainer_ddp.py','distributed_train.py','train_worker.py','bootstrap.sh','requirements.txt']
def compute_run_key(root):
 root=Path(root)
 hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in RUN_FILES}
 return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
