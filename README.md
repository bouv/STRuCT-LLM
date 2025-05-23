

# Evals for STRuCT-LLM

This is a reusable and easy-to-reproduce evals as seen in the paper.

## Get started

We rely on LangChain and vLLM for interacting with our models.

```
pip install -r requirements.txt
```

After installing, you can run the serving script which log to `output.log`:

```
HF_TOKEN=xyz sh serve_vllm.sh
```

Then the last thing we need is to setup credentials to access the local model and a remote judge:

```bash
export CAND_API_KEY=xyz
export CAND_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=sk-Xum8JoQUf2m5eon9z1snbw
export JUDGE_BASE_URL=https://api.marketplace.novo-genai.com/v1
```

## Running evals

We have implemented extensive evaluations. Here are the CLI options to choose from:

```
usage: Universal evaluation harness [-h] --task {cypher,qa,bird,clinton,crt_qa,mimic,tablebench} --model_path MODEL_PATH [--judge_model JUDGE_MODEL] [--workers WORKERS] [--base_dir BASE_DIR]

options:
  --task {cypher,qa,bird,clinton,crt_qa,mimic,tablebench}
                        Which benchmark to run
  --model_path MODEL_PATH
                        OpenAI compatible model to evaluate
  --judge_model JUDGE_MODEL
                        OpenAI compatible model to use as judge
  --workers WORKERS     Number of parallel samples to evaluate.
  --base_dir BASE_DIR   The base directory to load data from.
```

Cypher

```
python run_eval.py --task cypher --model_path mb618/Owen_14B_2tasks --judge_model openai_gpt41_nano
```

QA:

```
python run_eval.py --task qa --model_path mb618/Owen_14B_2tasks --use_prompt=final_answer --data=cr --score=EM
python run_eval.py --task qa --model_path mb618/Owen_14B_2tasks --use_prompt=final_answer --data=cr --score=in
```

Bird:

```
python run_eval.py --task bird --model_path mb618/Owen_14B_2tasks --use_evidence=True
```

Clinton:

```
python run_eval.py --task clinton --model_path mb618/Owen_14B_2tasks --judge_model openai_gpt41_nano --use_system=True
```

CRT QA:

```
python run_eval.py --task crt_qa --model_path mb618/Owen_14B_2tasks --use_prompt=final_answer --score=EM
python run_eval.py --task crt_qa --model_path mb618/Owen_14B_2tasks --use_prompt=final_answer --score=in
```

Mimic:

```
python run_eval.py --task mimic --model_path mb618/Owen_14B_2tasks
```

TableBench:

```
python run_eval.py --task tablebench --model_path mb618/Owen_14B_2tasks --score=EM
python run_eval.py --task tablebench --model_path mb618/Owen_14B_2tasks --score=in
```
