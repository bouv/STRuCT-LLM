import os
import torch
import numpy as np
import argparse
import wandb
from datasets import load_dataset, Dataset,concatenate_datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from rewards import (
    comp_ged_reward,
    llm_scoring_classes_judge_reward,
    string_matching_reward
)

np.random.seed(88)

def parse_args():
    parser = argparse.ArgumentParser(description='GRPO Training Arguments')

    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--load_in_4bit', type=bool, default=True)
    parser.add_argument('--reward_func', type=str, nargs='+', default=['comp_ged_reward' , 'llm_scoring_classes_judge_reward', 'string_matching_reward'])
    parser.add_argument('--model_name', type=str, default="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit")
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--num_generations', type=int, default=6)
    parser.add_argument('--reward_weights', type=float, nargs='+', default=[0.4,0.3,0.3])
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--per_device_train_batch_size', type=int, default=24)
    parser.add_argument('--output_dir', type=str, default='model_checkpoints_GRPO', help="Directory to save model checkpoints")
    parser.add_argument('--final_model_dir', type=str, default='final_models_GRPO', help="Directory to save final model")

    return parser.parse_args()

args = parse_args()

reward_names = args.reward_func if len(args.reward_func) < 4 else '4_rewards'
wandb_name = f"{args.model_name.replace('/', '_')}_{'_'.join(reward_names)}_{args.learning_rate}_{args.gradient_accumulation_steps}_{args.per_device_train_batch_size}_{args.lr_scheduler_type}_{'_'.join(map(str, args.reward_weights))}"
wandb_name = wandb_name.lower().replace(' ', '_')

wandb.login()
wandb.init(
    dir='wandb',
    name=wandb_name,
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=args.max_seq_length,
    dtype=args.dtype,
    load_in_4bit=args.load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=args.lora_rank,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

system_prompt= """A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: """

sql_prompt="""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags.

### Input:
Here is the relevant table info: {table_info}.
Write a SQLite query for the following task: {question}.

### Response:"""

cypher_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Cypher expert. Given an input question, create a syntactically correct Cypher query to run. Enclose the final query within "```cypher" and "```" tags.

### Input:
Here is the relevant graph information: {table_info}.
Write a Cypher query for the following task: {question}.
Assistant:"""




def formatting_prompts_func(examples, query_type):
    question = examples['question']
    table_info = examples['context']
    query = examples['answer']

    template = sql_prompt if query_type=='sql' else cypher_prompt

    text = template.format(
        table_info=table_info,
        question=question, 
    ) 
    text = [{'content':system_prompt, 
          'role': 'system'},
         {'content':text, 
          'role': 'user'}]
    
    return {'question':question , 'answer': query,"prompt": text,  'context': table_info,  'query_type': query_type}




# ################# get sql data ###############

loaded_dataset = load_dataset("xu3kev/BIRD-SQL-data-train", split="train").shuffle(seed=42)

dataset_train = loaded_dataset.select(range(0, int(0.95 * len(loaded_dataset))))  # 80% for training
dataset_val = loaded_dataset.select(range(int(0.95 * len(loaded_dataset)), int(1 * len(loaded_dataset))))  # 10% for validation

dataset_train = dataset_train.rename_columns({
    "schema": "context",
    "SQL": "answer"  
})


dataset_val = dataset_val.rename_columns({
    "schema": "context",
    "SQL": "answer"
})

train_dataset_sql = dataset_train.map(
    lambda example: formatting_prompts_func(example, 'sql'),
)

val_dataset=dataset_val.select(range(100))

val_dataset_sql = val_dataset.map(
    lambda example: formatting_prompts_func(example, 'sql'),
)


print(train_dataset_sql[0])

# ################# get cypher data ###############


loaded_dataset = load_dataset('neo4j/text2cypher-2024v1', split="train").shuffle(seed=42)
clinton_data = load_dataset("jls205/synthetic_cot_traces_cypher", data_files="Cypher_cot.csv")['train']
print(clinton_data)
clinton_questions = set(clinton_data['question'])
filtered_loaded_dataset = loaded_dataset.filter(lambda x: x['question'] not in clinton_questions)
print(filtered_loaded_dataset)

filtered_loaded_dataset = filtered_loaded_dataset.rename_column("schema", "context")
filtered_loaded_dataset = filtered_loaded_dataset.rename_column("cypher", "answer")

filtered_loaded_dataset=filtered_loaded_dataset.shuffle(seed=42)

dataset_train = filtered_loaded_dataset.select(range(0, int(0.90 * len(filtered_loaded_dataset))))  
dataset_val = filtered_loaded_dataset.select(range( int(0.90 * len(filtered_loaded_dataset)), int(0.95 * len(filtered_loaded_dataset)) )).shuffle(seed=42)
dataset_val = dataset_val.select(range(0, 100))  # Limit to 100 samples for validation


train_dataset_cypher = dataset_train.map(
    lambda example: formatting_prompts_func(example, 'cypher'),
)


val_dataset_cypher = dataset_val.map(
    lambda example: formatting_prompts_func(example, 'cypher'),
)
print(train_dataset_cypher[0])


################### combine datasets ################

    
train_size = min(len(train_dataset_cypher), len(train_dataset_sql))
val_size = min(len(val_dataset_cypher), len(val_dataset_sql))

train_dataset_sql = train_dataset_sql.shuffle(seed=42).select(range(train_size))
train_dataset_cypher = train_dataset_cypher.shuffle(seed=42).select(range(train_size))
val_dataset_sql = val_dataset_sql.shuffle(seed=42).select(range(val_size))
val_dataset_cypher = val_dataset_cypher.shuffle(seed=42).select(range(val_size))

train_dataset = concatenate_datasets([train_dataset_sql, train_dataset_cypher]).shuffle(seed=42)
val_dataset = concatenate_datasets([val_dataset_sql, val_dataset_cypher]).shuffle(seed=42)



function_mapping = {
    'string_matching_reward': string_matching_reward,
    'comp_ged_reward': comp_ged_reward,
    'llm_scoring_classes_judge_reward': llm_scoring_classes_judge_reward,
}

reward_funcs = [function_mapping[name] for name in args.reward_func if name in function_mapping]


max_prompt_length = 1024

training_args = GRPOConfig(
    learning_rate=args.learning_rate,
    output_dir=os.path.join(args.output_dir, wandb_name),
    lr_scheduler_type=args.lr_scheduler_type,
    optim="paged_adamw_8bit",
    logging_steps=1,
    logging_first_step=True,
    log_completions=True,
    reward_weights=args.reward_weights,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=24,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_generations=args.num_generations,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=50,
    eval_on_start=True,
    local_rank=-1,
    ddp_find_unused_parameters=False,
    max_prompt_length=max_prompt_length,
    max_completion_length=args.max_seq_length - max_prompt_length,
    save_steps=100,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

final_model_path = os.path.join(args.final_model_dir, wandb_name)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

wandb.finish()


