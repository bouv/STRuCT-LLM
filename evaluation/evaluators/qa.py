import json
from typing import Dict, Any
from argparse import ArgumentParser
from .base import BaseEvaluator
from utils.helpers import get_EM_score
from utils import prompts

class QAEvaluator(BaseEvaluator):
    NAME = "qa"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass
        
    def load_data(self):
        with open(f"{self.args.base_dir}/QA/CR-LT-QA.json") as file:
            data = json.load(file)
        
        return data 

    def evaluate_one(self, example) -> Dict[str, Any]:
        template, system = (prompts.QA_CR_FINAL_TEMPLATE, prompts.QA_CR_FINAL_SYSTEM)
        conversation = [
            {
                "content": system,
                "role": "system",
            },
            {
                "content": template.format(example["KG Triples"], example["query"]),
                "role": "user",
            },
        ]

        response = self.cand(conversation)
        final_answer = response.split("Final Answer:")[-1]
        
        final_answer = str(final_answer).lower()
        example["answer"] = str(example["answer"]).lower()

        if final_answer.replace(" ", "").lower() == "yes":
            final_answer = "True"
        elif final_answer.replace(" ", "").lower() == "no":
            final_answer = "False"
        


        EM_score = get_EM_score(example["answer"], final_answer)


        return {
            "question": example["query"],
            "llm": response,
            "answer": example["answer"],
            "scores": { "EM_score": EM_score,},
        }

