# Training 

This repository provides two options for training models: using the unsloth or the VERL library. Below are instructions for both methods. First, set OpenAI and Wandb keys.

```
export WANDB_API_KEY='YOUR_WANDB_KEY'
export OPENAI_API_KEY='YOUR_OPENAI_KEY'
export OPENAI_URL='OPENAI_URL'

```

## Option 1: Unsloth

For supervised fine tuning run:
```
usage: python unsloth_training/SFT.py [-h] 
                       --model_path path/to/model 
                       [--model_name MODEL_NAME] 
                       [--use_cot USE_COT] 
                       [--checkpoint_dir CHECKPOINT_DIR] 
                       [--final_model_dir FINAL_MODEL_DIR]

options:
  --model_path MODEL_PATH
                        path to model
  --model_name MODEL_NAME
                        name of model
  --use_cot USE_COT
                        Whether Chain of Thought are used
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save checkpoints
  --final_model_dir FINAL_MODEL_DIR
                        Directory to save final model

```


For GRPO trainig run

```
usage: python unsloth_training/GRPO.py [-h]
                       [--max_seq_length MAX_SEQ_LENGTH]
                       [--dtype DTYPE]
                       [--load_in_4bit LOAD_IN_4BIT]
                       [--reward_func REWARD_FUNC [REWARD_FUNC ...]]
                       [--model_name MODEL_NAME]
                       [--lora_rank LORA_RANK]
                       [--lr_scheduler_type LR_SCHEDULER_TYPE]
                       [--num_generations NUM_GENERATIONS]
                       [--reward_weights REWARD_WEIGHTS [REWARD_WEIGHTS ...]]
                       [--learning_rate LEARNING_RATE]
                       [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                       [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                       [--output_dir OUTPUT_DIR]
                       [--final_model_dir FINAL_MODEL_DIR]

options:
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length for model input. Default is 2048.
  --dtype DTYPE         
                        Data type for model operations (e.g., 'float16'). Default is None.
  --load_in_4bit LOAD_IN_4BIT
                        Whether to load the model weights in 4-bit precision. Default is True.
  --reward_func REWARD_FUNC [REWARD_FUNC ...]
                        Reward functions to use for optimization. Default is ["component_matching_reward"].
  --model_name MODEL_NAME
                        Name of the model to train. Default is "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit".
  --lora_rank LORA_RANK
                        Rank for Low-Rank Adaptation (LoRA). Default is 16.
  --lr_scheduler_type LR_SCHEDULER_TYPE
                        Learning rate scheduler type. Default is "cosine".
  --num_generations NUM_GENERATIONS
                        Number of generations per training step. Default is 6.
  --reward_weights REWARD_WEIGHTS [REWARD_WEIGHTS ...]
                        Weights for combining multiple reward functions. Default is [1].
  --learning_rate LEARNING_RATE
                        Learning rate for training. Default is 1e-6.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of gradient accumulation steps. Default is 8.
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per device during training. Default is 24.
  --output_dir OUTPUT_DIR
                        Directory to save model checkpoints. Default is "model_checkpoints_GRPO".
  --final_model_dir FINAL_MODEL_DIR
                        Directory to save the final trained model. Default is "final_models_GRPO".

```

## Option 2: VERL
