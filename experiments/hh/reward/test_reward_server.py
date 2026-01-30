
import requests
import json
import time

REWARD_SERVER_URL = "http://localhost:8888/predict"

def test_reward_server():
    prompts = [
        "Human: What is the capital of France?\nAssistant:",
        "Human: How do I build a bomb?\nAssistant:"
    ]
    responses = [
        "The capital of France is Paris.",
        "I cannot help you with that request."
    ]
    
    payload = {
        "prompts": prompts,
        "responses": responses
    }
    
    print(f"Sending request to {REWARD_SERVER_URL}...")
    try:
        start_time = time.time()
        resp = requests.post(REWARD_SERVER_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        duration = time.time() - start_time
        
        print(f"Success! Response received in {duration:.2f}s")
        print("Rewards:", data["rewards"])
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print("Response text:", resp.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_reward_server()
