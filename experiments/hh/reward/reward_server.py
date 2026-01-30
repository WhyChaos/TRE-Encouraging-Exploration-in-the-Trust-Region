from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import os
import gc
import math
from concurrent.futures import ThreadPoolExecutor

# Avoid tokenizer parallelism issues in forked processes/threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

# Global variables
models = []
tokenizer = None
devices = []

class BatchRewardRequest(BaseModel):
    prompts: list[str]
    responses: list[str]

class BatchRewardResponse(BaseModel):
    rewards: list[float]

MODEL_PATH = os.getenv(
    "HH_REWARD_MODEL_PATH",
    os.path.expanduser("~/model/Qwen/Qwen2.5-1.5B-Instruct-ultrafeedback_binarized-reward")
)

@app.on_event("startup")
async def startup_event():
    global models, tokenizer, devices
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This reward server requires a GPU.")
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Loading models on all GPUs...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    for i in range(num_gpus):
        device_name = f"cuda:{i}"
        print(f"Loading reward model replica on {device_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="flash_attention_2"
        )
        model.to(device_name)
        model.eval()
        models.append(model)
        devices.append(device_name)
        
    print(f"All {len(models)} reward models loaded successfully!")

def process_batch_chunk(gpu_id, prompts, responses):
    global models, tokenizer, devices
    
    if not prompts:
        return []
        
    device = devices[gpu_id]
    model = models[gpu_id]
    
    batch_messages = []
    for p, r in zip(prompts, responses):
        batch_messages.append([
            {"role": "user", "content": p},
            {"role": "assistant", "content": r}
        ])
    
    inputs = None
    outputs = None
    rewards = []

    try:
        # Tokenization
        texts = [tokenizer.apply_chat_template(
            msgs, 
            tokenize=False, 
            add_generation_prompt=False
        ) for msgs in batch_messages]
        
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Handle scalar output correctly
            if outputs.logits.shape == torch.Size([1]):
                 # Single element case, though usually batch
                 rewards = [outputs.logits.item()]
            else:
                 rewards = outputs.logits.squeeze(-1).tolist()
            
            # Ensure rewards is a list (squeeze on scalar might return float if 0-dim)
            if isinstance(rewards, float):
                 rewards = [rewards]
                 
        return rewards
    except Exception as e:
        print(f"Error processing chunk on {device}: {e}")
        raise e
    finally:
        if inputs is not None:
            del inputs
        if outputs is not None:
            del outputs

@app.post("/predict", response_model=BatchRewardResponse)
async def predict(request: BatchRewardRequest):
    global models
    
    try:
        total_len = len(request.prompts)
        if total_len == 0:
            return BatchRewardResponse(rewards=[])
            
        num_gpus = len(models)
        
        # If no GPUs loaded (should not happen due to check), or 1 GPU
        if num_gpus <= 1:
            return BatchRewardResponse(rewards=process_batch_chunk(0, request.prompts, request.responses))
            
        # Distribute across GPUs
        # We process in parallel using threads. 
        # Since we just do inference, GIL release in PyTorch C++ backend allows parallelism.
        
        chunk_size = math.ceil(total_len / num_gpus)
        chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_len)
            if start < end:
                chunks.append((i, request.prompts[start:end], request.responses[start:end]))
        
        results = []
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(process_batch_chunk, c[0], c[1], c[2]) for c in chunks]
            for f in futures:
                results.extend(f.result())
                
        return BatchRewardResponse(rewards=results)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    port = int(os.getenv("REWARD_SERVER_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
