import os
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset


os.environ["WANDB_PROJECT"] = "tre_hh_1.5b"

training_args = RewardConfig(
    output_dir="1.5b_reward_model/",
    report_to="wandb",
    run_name="1.5b_reward_model",
    model_init_kwargs={"num_labels": 1}, 
    num_train_epochs=3,
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=8,  
    learning_rate=1e-5,                      
    warmup_ratio=0.1,
    center_rewards_coefficient=1e-2, 
    bf16=True,
    max_length=None,  
)

trainer = RewardTrainer(
    model="../model/Qwen/Qwen2.5-1.5B-Instruct",
    args=training_args,
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()