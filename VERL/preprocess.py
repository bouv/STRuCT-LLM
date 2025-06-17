import os
import datasets
import argparse
from datasets import Dataset, load_dataset, interleave_datasets, concatenate_datasets
import pandas as pd
from transformers import AutoTokenizer


cypher_prompt ="""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Cypher expert. Given an input question, create a syntactically correct Cypher query to run. Enclose the final query within "```cypher" and "```" tags.

### Input:
Here is the relevant graph information: {}.
Write a Cypher query for the following task: {}.

### Response:"""


sql_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags.

### Input:
Here is the relevant table info: {}.
Write a SQLite query for the following task: {}.

### Response:"""

system_prompt= """A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default='/dcai/projects02/data/mbvk/mixed_cot_sql_cypher/')
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()
    clinton_data=pd.read_csv('path/correct_sql_qa_clinton.csv')
    clinton_data=Dataset.from_pandas(clinton_data)
    clinton_data=clinton_data.shuffle(seed=42)
    clinton_data = clinton_data.map(lambda x: {'query_type': 'sql'})
    train_size_1 = int(0.99 * len(clinton_data))
    test_size_1 = int(0.01 * len(clinton_data))
    train_dataset_clinton = clinton_data.select(range(train_size_1))
    val_dataset_clinton = clinton_data.select(range(train_size_1, len(clinton_data)))



    cypher_data = pd.read_csv('path/Cypher_cot.csv')  # Added for 
    cypher_data=Dataset.from_pandas(cypher_data)
    cypher_data.rename_column('answer','query')
    train_size_2 = int(0.99 * len(cypher_data))
    test_size_2 = int(0.01 * len(cypher_data))
    cypher_data=cypher_data.shuffle(seed=42)
    clinton_data = clinton_data.map(lambda x: {'query_type': 'cypher'})
    train_dataset_cypher = cypher_data.select(range(train_size_2))
    val_dataset_cypher = cypher_data.select(range(train_size_2, len(cypher_data)))


    train_size=min(train_size_2, train_size_1)
    test_size= min(test_size_2, test_size_1)

    train_dataset_cypher = train_dataset_cypher.select(range(train_size))
    val_dataset_cypher = val_dataset_cypher.select(range(test_size))
    train_dataset_clinton = train_dataset_clinton.select(range(train_size))
    val_dataset_clinton = val_dataset_clinton.select(range(test_size))

    train_combined = concatenate_datasets([train_dataset_cypher, train_dataset_clinton])
    val_combined = concatenate_datasets([val_dataset_cypher, val_dataset_clinton])

    # Shuffle the combined datasets
    train_combined = train_combined.shuffle(seed=42)
    val_combined = val_combined.shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct')
    EOS_TOKEN = tokenizer.eos_token
    print('marc')

    train_data = []
    test_data = []

    for example in train_combined:
        prompt_template = sql_prompt if example['query_type']=='sql' else cypher_prompt
        question = example['question']
        table_info = example['context']
        query = "```sql "+ example['answer'] + " ```"
        thinking = example['thinking'].replace('<thinking>','')
        thinking = '<think> ' + thinking.replace('</thinking>','') + ' </think>'
        train_data.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_template.format(table_info, question)},
                    {"role": "assistant", "content": thinking+query+ EOS_TOKEN},
                ]
            }
        )

    for example in val_combined:
        prompt_template = sql_prompt if example['query_type']=='sql' else cypher_prompt
        question = example['question']
        table_info = example['context']
        query = "```sql "+ example['answer'] + " ```"
        thinking = example['thinking'].replace('<thinking>','')
        thinking = '<think> ' + thinking.replace('</thinking>','') + ' </think>'
        test_data.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_template.format(table_info, question)},
                    {"role": "assistant", "content": thinking+query+ EOS_TOKEN},
                ]
            }
        )
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)


    train_df.to_parquet(os.path.join('path', "train_mixed_cot.parquet"))
    test_df.to_parquet(os.path.join('path', "test_mixed_cot.parquet"))

if __name__ == "__main__":
    main()
