COT_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: """


CYPHER_ALPACA = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
bird_qwen
### Instruction:
You are a Cypher expert. Given an input question, create a syntactically correct Cypher query to run. Enclose the final query within "```cypher" and "```" tags.

### Input:
Here is the relevant graph information: {}.
Write a Cypher query for the following task: {}.

### Response:"""



QA_CR_FINAL_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The answer should follow the format below:
<think> reasoning process here </think> Final Answer: Answer \nEnsure the final answer format is the last output line and can only be in the "Final Answer: Answer" form, no other form. Ensure the "Answer" is "Yes", "No", ord "I don't know". User: """

QA_CR_FINAL_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a graph reasoning expert. Below are facts in the form of the triple meaningful to answer the query.

### Input:
Triples: {}.
Please answer the query: {}.

### Response:"""



QA_CLAIMS_FINAL_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The answer should follow the format below:
<think> reasoning process here </think> Final Answer: Answer \nEnsure the final answer format is the last output line and can only be in the "Final Answer: Answer" form, no other form. Ensure the "Answer" is "True", "False", or "I don't know". User: """

QA_CLAIMS_FINAL_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a graph reasoning expert. Below are facts in the form of the triple meaningful to verify the query. Please reason step by step, and put your final answer within \\boxed{{}}.

### Input:
Triples: {}.
Please verify the query: {}.

### Response:"""




BIRD_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: 
"""

BIRD_QWEN_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags.

### Input:
Here is the relevant table info: {}.
Write a SQLite query for the following task: {}.

### Response:"""

BIRD_QWEN_TEMPLATE_EVIDENCE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags. 

### Input:
Here is the relevant table info: {}.
You can use this external knowledge for help: {}.
Write a SQLite query for the following task: {}.

### Response:"""




CRT_NORMAL_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. If the question is NOT annswerable given the table information answer with the word "Unanswerable" and nothing else. User: """

CRT_NORMAL_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a tabular reasoning expert. Read the table below to answer the following question. Encose you reasoning process in <think> and </think> tags and provide a clear and concise final answer after the </think> tag, if possible in only one word or one number rounded to two decimals. If the question is NOT annswerable given the table information answer with the word "Unanswerable" and nothing else.

### Input:
Table: {}.
Question: {}.

### Response:"""






MIMIC_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: """


MIMIC_QWEN_ALPACA_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. For binary questions the sql query should return 0 or 1. You can use 'now' in the query which will be replaced with the current time automatically. Enclose the final sql query within "```sql" and "```" tags.
If the question is not answerable return null and nothing else.

### Input:
Here is the relevant table info: {table_info}.
Write a SQLite query for the following task: {question}.

### Response:"""



TAB_QWEN_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
    The assistant first thinks about the reasoning process in the mind and then provides the user
    with the answer. The answer should follow the format below:
    <think> reasoning process here </think> Final Answer: AnswerName1, AnswerName2...\nEnsure the final answer format is the last output line and can only be in the "Final Answer: AnswerName1, AnswerName2..." form, no other form. Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. User: """

TAB_QWEN_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a table analyst. Your task is to answer questions based on the table content. The answer should follow the format below:
<think> reasoning process here </think> Final Answer: AnswerName1, AnswerName2...\nEnsure the final answer format is the last output line and can only be in the "Final Answer: AnswerName1, AnswerName2..." form, no other form. Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation.

### Input:
Read the table below in JSON format: {}.
Answer the following question: {}.

### Response:"""





SM3_ALPACA = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a {query_language} expert. Given an input question, create a syntactically correct {query_language} query to run. Enclose the final query within "```{query_language}" and "```" tags.
{extra_info}

### Input:
Here is the relevant graph information: {context}.
Write a {query_language} query for the following task: {question}.

### Response:"""



SM3_ALPACA_examples = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a {query_language} expert. Given an input question, create a syntactically correct {query_language} query to run. Enclose the final query within "```{query_language}" and "```" tags.
{extra_info}

### Input:
Here are some examples:
{examples}

Here is the relevant graph information: {context}.
Write a {query_language} query for the following task: {question}.

### Response:"""
