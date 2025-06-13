import pandas as pd
from typing import Dict, Any
from argparse import ArgumentParser
from datasets import load_dataset
from .base import BaseEvaluator
from utils.prompts import COT_SYSTEM, CYPHER_ALPACA
from utils.helpers import (
    extract_xml_answer_cypher,
    evaluate_generated_response,
    evaluate_bleu_response_manual,
    get_EM_score
)


class CypherEvaluator(BaseEvaluator):
    NAME = "cypher"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass

    def load_data(self):
        loaded_dataset = load_dataset(
            "neo4j/text2cypher-2024v1", split="train"
        ).shuffle(seed=42)
        cypher_cot_data = load_dataset("jls205/synthetic_cot_traces_clinton", data_files="cot.csv")['train'].shuffle(seed=42)
        cypher_questions = set(cypher_cot_data["question"])

        filtered_data = loaded_dataset.filter(
            lambda x: x["question"] not in cypher_questions
        )
        filtered_data = filtered_data.rename_column("schema", "context")
        filtered_data = filtered_data.rename_column("cypher", "answer")
        filtered_data = filtered_data.shuffle(seed=42)

        test_data = filtered_data.select(
            range(
                int(0.95 * len(filtered_data)),
                int(1 * len(filtered_data)),
            )
        ).shuffle(seed=42)

        return test_data

    def evaluate_one(self, example) -> Dict[str, Any]:
        conversation = [
            {"role": "system", "content": COT_SYSTEM},
            {
                "role": "user",
                "content": CYPHER_ALPACA.format(
                    example["context"], example["question"]
                ),
            },
        ]

        response = self.cand(conversation)
        final_answer = extract_xml_answer_cypher(response)
        final_answer=str(final_answer).lower()
        example["answer"]=str(example["answer"]).lower()
        

        sql_eval = [
            (
                "system",
                f"""
                    Evaluate the predicted cypher query against the correct cypher query (ie do they return the same result).
                    Predicted query: {final_answer.upper()}
                    Correct query: {example['answer'].upper()}
                    Return ONLY "Correct" or "Wrong"
                        """,
            ),
            (final_answer.upper()),
        ]
        score_llm = evaluate_generated_response(self.judge.chat, sql_eval)

        score_blue_manual=evaluate_bleu_response_manual(example["answer"], final_answer)
        EM_score=get_EM_score(example["answer"], final_answer)

        return {
            "question": example["question"],
            "llm": response,
            "answer": example["answer"],
            "scores": { "EM_score": EM_score, "llm_accuracy": score_llm, "BLEU_manual": score_blue_manual},
        }
