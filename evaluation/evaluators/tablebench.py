import re
from datasets import load_dataset
from typing import Dict, Any
from argparse import ArgumentParser
from .base import BaseEvaluator
from utils import prompts
from utils.helpers import get_EM_score, evaluate_rouge_response_manual


class TableBenchEvaluator(BaseEvaluator):
    NAME = "tablebench"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass

    def load_data(self):
        dataset = load_dataset("Hyeonsieun/Tablebench")
        dataset = (
            dataset["test"]
            .filter(lambda example: example["qtype"] == "FactChecking")
            .shuffle(seed=42)
        )

        return dataset

    def evaluate_one(self, example) -> Dict[str, Any]:
        system = prompts.TAB_QWEN_SYSTEM
        template = prompts.TAB_QWEN_TEMPLATE

        conversation = [
            {"content": system, "role": "system"},
            {
                "content": template.format(example["table"], example["question"]),
                "role": "user",
            },
        ]

        response = self.cand(conversation)
        final_answer = (
            response.split("Final Answer:")[-1]
            if len(response.split("Final Answer:")) > 1
            else ""
        )

        ex = re.findall(r"\d+\.\d+|\d+", example["answer"])
        ex = str(ex[0]) if len(ex) > 0 else example["answer"]

        EM_score = get_EM_score(ex, final_answer)
        rouge_manual=evaluate_rouge_response_manual( ex, final_answer)
        return {
            "question": example["question"],
            "llm": response,
            "answer": example["answer"],
            "scores": {'EM_score': EM_score, 'rouge score': rouge_manual},
        }
