import json
from datasets import Dataset
from typing import Dict, Any
from argparse import ArgumentParser
from .base import BaseEvaluator
from langchain_community.utilities import SQLDatabase
from utils.helpers import (
    extract_table_names,
    get_table_specific_info,
    post_process_sql,
    extract_xml_answer_sql,
    evaluate_single_prediction,
)
from utils.prompts import (
    MIMIC_SYSTEM,
    MIMIC_QWEN_ALPACA_TEMPLATE,
)


class MimicEvaluator(BaseEvaluator):
    NAME = "mimic"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass

    def load_data(self):
        self.db_path = f"{self.args.base_dir}/EHRSQL/dataset/ehrsql/mimic_iii/mimic_iii.sqlite"
        db = SQLDatabase.from_uri(
            f"sqlite:///{self.db_path}", sample_rows_in_table_info=3
        )
        self.full_table_info = db.get_context()["table_info"]

        with open(f"{self.args.base_dir}/EHRSQL/dataset/ehrsql/mimic_iii/test.json") as f:
            test = json.load(f)

        test_data = [
            {"question": item["question"], "query": item["query"]} for item in test
        ]
        test_dataset = Dataset.from_list(test_data)

        return test_dataset
        
    def evaluate_one(self, example) -> Dict[str, Any]:
        relevant_tables = extract_table_names(example["query"])

        specific_table_info = " "
        for table in relevant_tables:
            try:
                specific_table_info = (
                    specific_table_info
                    + "Information for Table "
                    + table
                    + ": "
                    + get_table_specific_info(self.full_table_info, [table])
                    + "\n"
                )
            except Exception as e:
                specific_table_info += f"Information for Table {table}: No detailed information available\n\n"

        template = MIMIC_QWEN_ALPACA_TEMPLATE

        conversation = [
            {"content": MIMIC_SYSTEM, "role": "system"},
            {
                "content": template.format(
                    table_info=specific_table_info,
                    question=example["question"],
                ),
                "role": "user",
            },
        ]

        response = self.cand(conversation)
        pred_query = post_process_sql(extract_xml_answer_sql(response))
        truth_query = post_process_sql(example["query"])
        

        ans_real, ans_pred = evaluate_single_prediction(
            truth_query, pred_query, 600, self.db_path
        )
        score = 1 if ans_real == ans_pred else 0
        score_2 = 1 if set(ans_real) == set(ans_pred) else 0
        truth_is_null = example["query"].strip().lower() == "null"
        llm_said_null = response.strip().lower().startswith("null")


        precision_ans=None
        precision_exec=None
        recall_ans=None
        recall_exec=None
        if ans_pred!='null':
            precision_ans=1 if ans_real != 'null' else 0
            precision_exec=1 if set(ans_real) == set(ans_pred) else 0
        if ans_real!='null':
            recall_ans=1 if ans_pred != 'null' else 0
            recall_exec=1 if set(ans_real) == set(ans_pred) else 0

        scores = {}
        scores['accuracy']=score
        scores['acc2']=score_2
        if precision_ans is not None:
            scores["precision_ans"] = precision_ans
        if precision_exec is not None:
            scores["precision_exec"] = precision_exec
        if recall_ans is not None:
            scores["recall_ans"] = recall_ans
        if recall_exec is not None:
            scores["recall_exec"] = recall_exec


        return {
            "question": example["question"],
            "llm": response,
            "answer": example["query"],
            "scores": scores,
        }

