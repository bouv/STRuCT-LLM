import pandas as pd
from typing import Dict, Any
from datasets import Dataset
from argparse import ArgumentParser
from langchain_community.utilities import SQLDatabase
import logging
from .base import BaseEvaluator
from utils.helpers import execute_sqlite, extract_xml_answer_sql, get_EM_score
from utils import prompts

class BirdEvaluator(BaseEvaluator):
    NAME = "bird"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        p.add_argument(
            "--use_evidence",
            type=str,
            default='false',
            required=True,
            help="Use evidence",
        )


    def load_data(self) -> Dataset:
        """
        1. Read the json exported from MiniDev.
        2. Build a hf-Dataset.
        3. Collect table-info for every db_id once, not per row.
        4. Attach the corresponding context string to every example.
        """
        json_path = f"{self.args.base_dir}/MINIDEV/mini_dev_sqlite.json"
        df = pd.read_json(json_path)
        ds = Dataset.from_pandas(df)

        self.context_per_db: Dict[str, str] = {}
        for db_id in set(ds["db_id"]):
            sqlite_file = (
                f"{self.args.base_dir}/MINIDEV/" f"dev_databases/{db_id}/{db_id}.sqlite"
            )
            db = SQLDatabase.from_uri(
                f"sqlite:///{sqlite_file}", sample_rows_in_table_info=0
            )
            self.context_per_db[db_id] = db.get_context()["table_info"]

        def attach_context(example):
            example["context"] = self.context_per_db[example["db_id"]]
            return example

        ds = (
            ds.rename_columns({"SQL": "answer"}) 
            .shuffle(seed=42)
            .map(attach_context)
        )
        return ds

    def evaluate_one(self, example) -> Dict[str, Any]:
        system, template = (
                prompts.BIRD_SYSTEM,
                prompts.BIRD_QWEN_TEMPLATE,
            )

        if self.args.use_evidence=='true':
            conversation = [
                {"content": system, "role": "system"},
                {
                    "content": prompts.BIRD_QWEN_TEMPLATE_EVIDENCE.format(
                        example["context"], example["evidence"], example["question"]
                    ),
                    "role": "user",
                },
            ]
        else:
            conversation = [
                {"content": system, "role": "system"},
                {
                    "content": template.format(example["context"], example["question"]),
                    "role": "user",
                },
            ]

        response = self.cand(conversation)
        

        pred_query = extract_xml_answer_sql(response)


        sqlite_file = (
            f"{self.args.base_dir}/MINIDEV/"
            f"dev_databases/{example['db_id']}/{example['db_id']}.sqlite"
        )

        try:
            correct_results = execute_sqlite(example["answer"], sqlite_file)
        except Exception as e:
            correct_results = f"Error executing correct SQL: {e}"
            return 
        if correct_results == None or len(correct_results) == 0:
            correct_results = "No results returned for the correct SQL query."
            return

        try:
            pred_results = execute_sqlite(pred_query, sqlite_file)
        except Exception as e:
            pred_results = f"Error executing predicted SQL: {e}"

        results_match = set(pred_results) == set(correct_results)
        EM_score=get_EM_score(example["answer"], pred_query)


        return {
            "question": example["question"],
            "llm": response,
            "answer": example["answer"],
            "scores": {"execution_accuracy": results_match, "EM_score": EM_score},
        }

