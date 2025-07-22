import pandas as pd
from typing import Dict, Any
from datasets import Dataset
from argparse import ArgumentParser
from langchain_community.utilities import SQLDatabase
import json
from .base import BaseEvaluator
from utils.helpers import execute_sqlite, extract_xml_answer_sql, get_EM_score
from utils import prompts

class SpiderEvaluator(BaseEvaluator):
    NAME = "spider"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass


    def load_data(self) -> Dataset:
        """
        1. Read the json exported from Spider_test.
        2. Build a hf-Dataset.
        3. Collect table-info for every db_id once, not per row.
        4. Attach the corresponding context string to every example.
        5. Filter out examples with None context.
        """
        # A) read the raw json ------------------------------------------------
        json_path = f"{self.args.base_dir}/spider_data/test.json"
        with open(json_path, "r") as file:
            data = json.load(file)
        dict_data = {}
        for key in data[0].keys():
            dict_data[key] = [
                json.dumps(item[key]) if isinstance(item[key], (dict, list)) else item[key]
                for item in data
            ]
        ds = Dataset.from_dict(dict_data)

        context_per_db: Dict[str, str] = {}
        valid_db_ids = set()  
        for db_id in set(ds["db_id"]):
            try:
                sqlite_file = sqlite_file = (
                    f"{self.args.base_dir}/spider_data/test_database/{db_id}/{db_id}.sqlite"
                )
                db = SQLDatabase.from_uri(
                    f"sqlite:///{sqlite_file}", sample_rows_in_table_info=0
                )
                context = db.get_context()["table_info"]
                if context is not None:  
                    context_per_db[db_id] = context
                    valid_db_ids.add(db_id)
            except Exception as e:
                print(f"An error occurred while processing {db_id}: {e}")

        def attach_context(example):
            db_id = example["db_id"]
            if db_id in context_per_db:
                example["context"] = context_per_db[db_id]
                example["valid"] = True
            else:
                example["context"] = None
                example["valid"] = False
            return example

        ds = (
            ds.shuffle(seed=42)
            .map(attach_context)
            .filter(lambda x: x["valid"])  
            .remove_columns(["valid"])  
        )


        return ds



    def evaluate_one(self, example) -> Dict[str, Any]:
        system, template = (
                prompts.BIRD_SYSTEM,
                prompts.BIRD_QWEN_TEMPLATE,
            )
        context=example['context']

        conversation = [
            {"content": system, "role": "system"},
            {"content": template.format( context, example["question"]),"role": "user",},
        ]

        response = self.cand(conversation)
        pred_query = extract_xml_answer_sql(response)

        sqlite_file = (
            f"{self.args.base_dir}/spider_data/"
            f"test_database/{example['db_id']}/{example['db_id']}.sqlite"
        )

        try:
            correct_results = execute_sqlite(example["query"], sqlite_file)
        except Exception as e:
            correct_results = f"Error executing correct SQL: {e}"
            return 
        if correct_results == None:
            correct_results = "No results returned for the correct SQL query."
            return

        try:
            pred_results = execute_sqlite(pred_query, sqlite_file)
        except Exception as e:
            pred_results = f"Error executing predicted SQL: {e}"
        

        results_match = set(pred_results) == set(correct_results)

        return {
            "question": example["question"],
            "llm": response,
            "answer": example["query"],
            "scores": { "execution_accuracy": results_match},
        }
