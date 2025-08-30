
# src/plant/inference.py
import numpy as np
import torch

@torch.no_grad()
def embed_sequences(model, dataloader, use_fp16=True):
    model.eval()
    device = next(model.parameters()).device
    outs = []
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_fp16 and torch.cuda.is_available()) else nullcontext()
    with ctx:
        for batch in dataloader:
            iv = batch["input_ids_virus"].to(device)
            amv = batch["attention_mask_virus"].to(device)
            outputs = model(input_ids_virus=iv, attention_mask_virus=amv)
            outs.append(outputs.hidden_state_virus.float().cpu().numpy())
    return np.concatenate(outs, axis=0)

from contextlib import nullcontext



