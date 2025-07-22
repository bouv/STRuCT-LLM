import os
import json
import random
import pandas as pd
from typing import Dict, Any
from argparse import ArgumentParser
from .base import BaseEvaluator
from utils.helpers import get_EM_score
import logging
from utils import prompts


class CRTQAEvaluator(BaseEvaluator):
    NAME = "crt_qa"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass

    def load_data(self):
        csv_folder_path = f"{self.args.base_dir}/CRT-QA/CRT-QA/CRT-QA/all_csv"

        with open(f"{self.args.base_dir}/CRT-QA/CRT-QA/CRT-QA/dataset.json") as file:
            dataset_json = json.load(file)

        all_tasks = []

        for file_name in dataset_json:
            csv_file_path = f"{csv_folder_path}/{file_name}"
            if os.path.exists(csv_file_path):
                csv_data = pd.read_csv(csv_file_path, sep="#")
                for question in dataset_json[file_name]:
                    question_text = question["Question name"]
                    answer = question["Answer"]
                    title = question["Tittle"]
                    all_tasks.append((title, question_text, csv_data, answer))

        with open(f"{self.args.base_dir}/CRT-QA/CRT-QA/CRT-QA/unanswerable.json") as file:
            unanswerable_json = json.load(file)
        for task in unanswerable_json:
            csv_file_path = f"{csv_folder_path}/{task['csv file']}"
            if os.path.exists(csv_file_path):
                csv_data = pd.read_csv(csv_file_path, sep="#")
                question_text = task["Question name"]
                answer = 'Unanswerable'
                all_tasks.append(('the table', question_text, csv_data, answer))

        random.Random(42).shuffle(all_tasks)

        return all_tasks

    def evaluate_one(self, example) -> Dict[str, Any]:
        system, template = (
                prompts.CRT_NORMAL_SYSTEM,
                prompts.CRT_NORMAL_TEMPLATE,
            )
        title, question_text, csv_data, answer = example

        conversation = [
            {"content": system, "role": "system"},
            {
                "content": template.format(
                    csv_data.to_json(orient="records", lines=False),
                    question_text,
                ),
                "role": "user",
            },
        ]

        response = self.cand(conversation)

        final_answer = response.split("</think>")[-1]

        EM_score = get_EM_score(answer, final_answer)
        score_in = 1 if str(answer).lower() in str(final_answer).lower() else 0
        
        precision_ans=None
        precision_exec=None
        recall_ans=None
        recall_exec=None
        if final_answer.lower().strip()!='unanswerable':
            precision_ans=1 if answer != 'Unanswerable' else 0
            precision_exec=EM_score
            logging.info(f"LLM {final_answer} truth {answer} llm answered {precision_ans} EM  {EM_score}")
        else:
            logging.info(f"LLM {final_answer} truth {answer} llm didnt answer EM  {EM_score}")
        if answer!='Unanswerable':
            recall_ans=1 if final_answer.lower().strip() != 'unanswerable' else 0
            recall_exec=EM_score
            logging.info(f"LLM {final_answer} truth {answer} true is answerable {recall_ans} EM  {EM_score}")
        else:
            logging.info(f"LLM {final_answer} truth {answer} true is unanswerable EM  {EM_score}")


        scores = {}
        scores['EM_score']=EM_score
        scores['score_in']=score_in
        if precision_ans is not None:
            scores["precision_ans"] = precision_ans
        if precision_exec is not None:
            scores["precision_exec"] = precision_exec
        if recall_ans is not None:
            scores["recall_ans"] = recall_ans
        if recall_exec is not None:
            scores["recall_exec"] = recall_exec

        return {
            "question": question_text,
            "llm": response,
            "answer": answer,
            "scores": scores,
        }

