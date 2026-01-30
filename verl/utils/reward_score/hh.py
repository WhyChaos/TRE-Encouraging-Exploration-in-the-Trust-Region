"""
HH (Helpful and Harmless) dataset reward function using external reward model.
"""

import os
import requests
import random
import time

# Service configuration
REWARD_SERVER_URL = os.getenv("REWARD_SERVER_URL", "http://localhost:8888/predict")


def get_reward_score(prompt, response, model_path=None):
    """
    Call the external reward model service
    """
    # Fallback to single item list for backward compatibility if needed, 
    # but the server now expects lists. 
    # Since we are updating the server to batch only, we must wrap single request in list.
    return get_reward_score_batch([prompt], [response], model_path)[0]

def get_reward_score_batch(prompts, responses, model_path=None):
    """
    Call the external reward model service with batch
    """
    # Process in chunks to avoid OOM on the reward server
    BATCH_SIZE = 256
    all_rewards = []
    
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        batch_responses = responses[i : i + BATCH_SIZE]
        
        payload = {
            "prompts": batch_prompts,
            "responses": batch_responses
        }
        
        try:
            resp = requests.post(REWARD_SERVER_URL, json=payload)
            if resp.status_code != 200:
                print(f"[HH Reward] Server error details: {resp.text}")
            resp.raise_for_status()
            all_rewards.extend(resp.json()["rewards"])
        except Exception as e:
            print(f"[HH Reward] Error calling reward service for batch {i}-{i+len(batch_prompts)}: {e}")
            raise e
            
    return all_rewards


def compute_score(solution_str, ground_truth, model_path=None):
    # Backward compatibility wrapper
    # If the caller uses single item, we wrap it.
    # But if we change NaiveRewardManager to call a new batch function, we can leave this as is
    # wrapped around batch calls.
    
    try:
        prompt = ground_truth["prompt"]
    except:
        raise ValueError("ground_truth must contain 'prompt' key for HH reward computation.")
    
    response = solution_str

    do_print = random.randint(1, 512) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

    retry_count = 0
    while True:
        try:
            return get_reward_score(prompt, response, model_path)
        except Exception as e:
            retry_count += 1
            print(f"[HH Reward] Request failed (attempt {retry_count}), retrying... Error: {e}")
            time.sleep(1)

def compute_score_batch(solution_strs, ground_truths, model_path=None):
    """
    Compute scores for a batch of solutions and ground truths.
    """
    prompts = [gt["prompt"] for gt in ground_truths]
    responses = solution_strs
    
    if len(prompts) > 0:
        print(f"--------------------------------")
        print(f"Prompt[0]: {prompts[0]}")
        print(f"Response[0]: {responses[0]}")
        print(f"Batch size: {len(prompts)}")
        
    retry_count = 0
    while True:
        try:
            return get_reward_score_batch(prompts, responses, model_path)
        except Exception as e:
            retry_count += 1
            print(f"[HH Reward] Batch request failed (attempt {retry_count}), retrying... Error: {e}")
            time.sleep(1)


