from typing import Dict, Any
from argparse import ArgumentParser
from datasets import Dataset
from .base import BaseEvaluator
from utils.helpers import (
    evaluate_generated_response,
    extract_SM3_answer,
)
import pandas as pd


class SM3Evaluator(BaseEvaluator):
    NAME = "SM3_examples"
    
    @classmethod
    def add_args(cls, p: ArgumentParser):
        p.add_argument(
            "--query_language",
            required=True,
            choices=["MQL", "SQL", "Cypher"],
            help="Specify the query language to be used for evaluation"
        )

    def load_data(self):
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



        sql_examples="""
            Question: Which encounter is related to allergy Animal dander (substance)?
            SQL query: ```sql SELECT DISTINCT e.description FROM encounters e LEFT JOIN allergies a ON a.encounter = e.id WHERE a.description=' Animal dander (substance)';```
            Question: Provide the list of patients associated with the payer Dual Eligible.
            SQL query: ```sql SELECT DISTINCT p.first, p.last FROM payers py LEFT JOIN payer_transitions pt ON py.id=pt.payer LEFT JOIN patients p ON pt.patient=p.id WHERE py.id='Dual Eligible';```
            Question: Give me the organization affiliated with the provider with the ID beff794b-089c-3098-9bed-5cc458acbc05.
            SQL query: ```sql SELECT org.name FROM providers pr LEFT JOIN organizations org ON pr.organization=org.id WHERE id='beff794b-089c-3098-9bed-5cc458acbc05';```
            Question: What is the base cost of medication with the code 205923.
            SQL query: ```sql SELECT DISTINCT base_cost FROM medications WHERE code='205923';```
            Question: What is the procedure code of the claim transaction 210ae4cd-7ca0-7da4-66a7-ef20b4f5db4d?
            SQL query: ```sql SELECT procedurecode FROM claims_transactions WHERE id='210ae4cd-7ca0-7da4-66a7-ef20b4f5db4d';```
            """

        cypher_examples="""
            Question: Which encounter is related to allergy Animal dander (substance)?
            Cypher query: ```cypher MATCH (e:Encounter)-[:HAS_DIAGNOSED]->(a:Allergy {{description: 'Animal dander (substance)'}}) RETURN DISTINCT e.description;```
            Question: Provide the list of patients associated with the payer Dual Eligible.
            Cypher query: ```cypher MATCH (p:Patient)-[:INSURANCE_START]->(py:Payer {{name: 'Dual Eligible'}}) RETURN DISTINCT p.firstName, p.lastName;```
            Question: Give me the organization affiliated with the provider with the ID beff794b-089c-3098-9bed-5cc458acbc05.
            Cypher query: ```cypher MATCH (o:Organization)-[:IS_PERFORMED_AT]->(p:Provider {{id: 'beff794b-089c-3098-9bed-5cc458acbc05'}}) RETURN o.name;```
            Question: What is the base cost of medication with the code 205923.
            Cypher query: ```cypher MATCH (m:Medication {{code: '205923'}}) RETURN m.baseCost;```
            Question: What is the procedure code of the claim transaction 210ae4cd-7ca0-7da4-66a7-ef20b4f5db4d?
            Cypher query: ```cypher MATCH (ct:ClaimTransaction {{id: '210ae4cd-7ca0-7da4-66a7-ef20b4f5db4d'}}) RETURN ct.procedureCode;```
        """

        mql_examples= """
            Question: Which encounter is related to allergy Animal dander (substance)?
            MongoDB query: ```mongodb db.patients.aggregate([ { $match: {"ENCOUNTERS.ALLERGIES.DESCRIPTION": "Animal dander (substance)"} }, { $unwind: "$ENCOUNTERS" }, { $unwind: "$ENCOUNTERS.ALLERGIES" }, { $match: {"ENCOUNTERS.ALLERGIES.DESCRIPTION": "Animal dander (substance)"} }, { $group: {_id: "$ENCOUNTERS.DESCRIPTION"} }, { $project: { _id: 0, encounter_description: "$_id" } } ])```
            Question: Provide the list of patients associated with the payer Dual Eligible.
            MongoDB query: ```mongodb db.patients.aggregate([ { $lookup: { from: "payers", localField: "PAYER_TRANSITIONS.PAYER_REF", foreignField: "PAYER_ID", as: "payer_details" } }, { $unwind: "$PAYER_TRANSITIONS" }, { $unwind: "$payer_details" }, { $match: { "payer_details.NAME": "Dual Eligible" } }, { $project: { _id: 0, first: "$FIRST", last: "$LAST" } }, { $group: {_ id: { first: "$first", last: "$last" } } }, { $project: { _id: 0, first: "$_id.first", last: "$_id.last" } }]);```
            Question: Give me the organization affiliated with the provider with the ID beff794b-089c-3098-9bed-5cc458acbc05.
            MongoDB query: ```mongodb db.providers.aggregate([{$match: {"PROVIDER_ID": "beff794b-089c-3098-9bed-5cc458acbc05"}},{$lookup: {from: "organizations",localField: "ORGANIZATION_REF",foreignField: "ORGANIZATION_ID",as: "organization"}},{$unwind: "$organization"},{$project: {_id: 0,organization_name: "$organization.NAME"}}])```
            Question: What is the base cost of medication with the code 205923.
            MongoDB query: ```mongodb db.patients.aggregate([ { $match: {"ENCOUNTERS.MEDICATIONS.CODE": 205923} }, { $unwind: "$ENCOUNTERS" }, { $unwind: "$ENCOUNTERS.MEDICATIONS" }, { $match: {"ENCOUNTERS.MEDICATIONS.CODE": 205923} }, { $project: { _id: 0, base_cost: "$ENCOUNTERS.MEDICATIONS.BASE_COST" } }])```
            Question: What is the procedure code of the claim transaction 210ae4cd-7ca0-7da4-66a7-ef20b4f5db4d?
            MongoDB query: ```mongodb db.patients.aggregate([ { $match: { "CLAIMS.CLAIM_TRANSACTIONS.CLAIM_TRANSACTION_ID": "210ae4cd-7ca0-7da4-66a7-ef20b4f5db4d" } }, { $unwind: "$CLAIMS" }, { $unwind: "$CLAIMS.CLAIM_TRANSACTIONS" }, { $match: { "CLAIMS.CLAIM_TRANSACTIONS.CLAIM_TRANSACTION_ID": "210ae4cd-7ca0-7da4-66a7-ef20b4f5db4d" } }, { $project: { _id: 0, procedure_code: "$CLAIMS.CLAIM_TRANSACTIONS.PROCEDURE_CODE" } }]);```
        """


        if self.args.query_language == "MQL":
            examples = mql_examples
        elif self.args.query_language == "Cypher":
            examples = cypher_examples
        else:
            examples = sql_examples
        
        template = prompts.SM3_ALPACA_examples
        schema = schemas.get(self.args.query_language)
        conversation = [
            {"content": prompts.COT_SYSTEM, "role": "system"},
            {
                "content": template.format(                
                query_language=query_languages.get(self.args.query_language),
                context=schema,
                question=example["question"],
                extra_info=extra_info,
                examples=examples
                ),
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


