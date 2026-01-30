#!/bin/bash

# Configuration
export REWARD_SERVER_PORT=8888
export HH_REWARD_MODEL_PATH="$HOME/model/reward_model"  # Path to your reward model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Specify which GPU to use for reward server

# Start the reward server in background
echo "Starting reward server on port $REWARD_SERVER_PORT..."
nohup python3 experiments/hh/reward/reward_server.py > reward_server.log 2>&1 &
SERVER_PID=$!

echo "Reward server started with PID $SERVER_PID"
echo "Waiting for server to be ready..."

# Wait for server to be ready
max_retries=50
count=0
while ! curl -s "http://localhost:$REWARD_SERVER_PORT/docs" > /dev/null; do
    sleep 2
    count=$((count+1))
    if [ $count -ge $max_retries ]; then
        echo "Error: Server failed to start within timeout"
        cat reward_server.log
        kill $SERVER_PID
        exit 1
    fi
    echo -n "."
done
echo "Server is ready!"

# Export URL for the training script
export REWARD_SERVER_URL="http://localhost:$REWARD_SERVER_PORT/predict"


echo "Environment setup complete. You can now run your training script."
echo "Running with REWARD_SERVER_URL=$REWARD_SERVER_URL"


