from typing import Dict, Any
from argparse import ArgumentParser
from datasets import Dataset
from .base import BaseEvaluator
from utils.helpers import (
    evaluate_generated_response,
    extract_SM3_answer,
)
import pandas as pd
from datasets import Dataset


class SM3Evaluator(BaseEvaluator):
    NAME = "SM3"
    
    @classmethod
    def add_args(cls, p: ArgumentParser):
        p.add_argument(
            "--query_language",
            required=True,
            choices=["MQL", "SQL", "Cypher"],
            help="Specify the query language to be used for evaluation"
        )

    def load_data(self) -> Dataset:
        """
        Loads the dataset and returns a Dataset object with question and answer pairs
        based on the selected query language.
        """
        
        df = pd.read_csv(f"{self.args.base_dir}/SM3/data/processed_data/processed_dev.csv")

        query_column_mapping = {
            "SQL": "sql",
            "Cypher": "cypher",
            "MQL": "mql"
        }
        
        answer_column = query_column_mapping[self.args.query_language]
        df_filtered = df[["question", answer_column]].rename(columns={answer_column: "answer"})
        dataset = Dataset.from_pandas(df_filtered)
        
        return dataset

    

    def evaluate_one(self, example) -> Dict[str, Any]:
        from utils import prompts, schemas
        schemas = {
            "SQL": schemas.sql_schema,
            "MQL": schemas.mql_schema,
            "Cypher": schemas.cypher_schema,
        }
        query_languages = {
            "SQL": "sql",
            "MQL": "mongodb",
            "Cypher": "cypher",
        }
        
        mql_info="""The "_id" field is only used as internal MomgoDB ObjectID and not as the domain specific ID of the objects in the collections. The objects are identified with a UUID in fields following a structure like PATIENT_ID, TRANSACTION_ID, CLAIM_ID..."""



        if self.args.query_language == "MQL":
            extra_info = mql_info
        else:
            extra_info = ""

        template = prompts.SM3_ALPACA
        schema = schemas.get(self.args.query_language)
        prompt= template.format(                
                query_language=query_languages.get(self.args.query_language),
                context=schema,
                question=example["question"],
                extra_info=extra_info
                )
        conversation = [
            {"content": prompts.COT_SYSTEM,
            "role": "system"},
            {
                "content": prompt,
                "role": "user",
            },
        ]

        response = self.cand(conversation)
        pred_query = extract_SM3_answer(response, self.args.query_language) 
        
        query_eval = [
            (
                "system",
                f"""
                    You are a {self.args.query_language} expert and your task is to evaluate if the predicted {self.args.query_language} query is correct 
                    based on the Schema and the correct {self.args.query_language} query. If no {self.args.query_language} query was found then the answer is Wrong. 
                    The query is considered correct even if the only mistakes are in letter casing (uppercase vs lowercase).
                    Schema: {schema}
                    Predicted query: {pred_query}
                    Correct {self.args.query_language} query: {example['answer']}
                    Return ONLY "Correct" or "Wrong"
                        """,
            ),
            (pred_query),
        ]
        score = evaluate_generated_response(self.judge.chat, query_eval)
        return {
            "question": example["question"],
            "llm": response,
            "answer": example["answer"],
            "scores": {"llm_accuracy": score},
        }


