# Evals for STRuCT-LLM

This is a reusable and easy-to-reproduce evals as seen in the paper.

## Get started

First, download the BIRD minidev data and Spider data from:

- https://yale-lily.github.io/spider (put the test dataset in data/spider_data/test.json)
- https://bird-bench.github.io/ (put the data in data/MINIDEV/mini_dev_sqlite.json)
- https://physionet.org/content/mimiciii/1.4/ (the path to this dataset is referred to path_to_mimic_iii_csv_files below)

and put them into the 'STRuCT-LLM/evaluation/data' folder.

Download the other datasets:

```
git clone https://github.com/D3Mlab/cr-lt-kgqa.git data/QA
git clone https://github.com/glee4810/EHRSQL.git data/EHRSQL
git clone https://github.com/jf87/SM3-Text-to-Query.git data/SM3
git clone https://github.com/zzh-SJTU/CRT-QA.git data/CRT-QA
cd data/CRT-QA
unzip CRT-QA.zip -d CRT-QA
cd ../EHRSQL/preprocess
python preprocess_db.py --data_dir <path_to_mimic_iii_csv_files> --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1

```

Then run the model serving script which log to `output.log`:

```
HF_TOKEN=xyz sh serve_vllm.sh /path/to/model MAX_MODEL_LENGTH
```


Then the last thing we need is to setup credentials to access the local model and a remote judge:

```bash
export CAND_API_KEY=xyz (huggingface token)
export CAND_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=xyz (OpenAI token)
export JUDGE_BASE_URL=xyz (OpenAI url)
```


## Running evals

We have implemented extensive evaluations. Here are the CLI options to choose from:

```
usage: Universal evaluation harness [-h] --task {cypher,qa,bird,clinton,crt_qa,mimic,tablebench} --model_path MODEL_PATH [--judge_model JUDGE_MODEL] [--workers WORKERS] [--base_dir BASE_DIR]

options:
  --task {cypher,qa,bird,crt_qa,mimic,tablebench, spider, mimic}
                        Which benchmark to run (text-to-cypher, LT-CRTKGQA, BIRD minidev, CRT-QA, Tablebench, spider, EHRSQL}
  --model_path MODEL_PATH
                        OpenAI compatible model to evaluate
  --judge_model JUDGE_MODEL
                        OpenAI compatible model to use as judge
  --workers WORKERS     Number of parallel samples to evaluate.
  --base_dir BASE_DIR   The base directory to load data from.
```
