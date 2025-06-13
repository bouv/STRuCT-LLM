import re
import sys
import json
import time
import sqlite3
import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from func_timeout import func_timeout, FunctionTimedOut
from langchain.output_parsers import PydanticOutputParser
from collections import defaultdict
from typing import List, Dict
import logging
from math import exp, log
from collections import defaultdict
from typing import List, Dict



        
def execute_sqlite(
    sql: str, db_path: str, timeout_s: float = 10.0, instruction_chunk: int = 1_000
):
    start = time.time()

    def _progress_handler():
        if time.time() - start > timeout_s:
            return 1
        return 0

    uri = f"file:{db_path}"
    with sqlite3.connect(uri, uri=True) as con:
        con.set_progress_handler(_progress_handler, instruction_chunk)
        cur = con.cursor()
        return cur.execute(sql).fetchall()


def extract_table_names(sql_query):
    """Extract table names from a SQL query, including nested queries."""
    sql_lower = sql_query.lower()
    sql_lower = re.sub(r"--.*$", "", sql_lower, flags=re.MULTILINE)

    def process_subquery(query):
        tables = set()
        from_splits = re.findall(r"\bfrom\s+([^\s,();]+)", query)
        tables.update(from_splits)

        join_splits = re.findall(r"\bjoin\s+([^\s,();]+)", query)
        tables.update(join_splits)

        in_clauses = re.findall(r"\bin\s*\((.*?)\)", query, re.DOTALL)
        for subquery in in_clauses:
            tables.update(process_subquery(subquery))
        return tables

    tables = process_subquery(sql_lower)

    cleaned_tables = set()
    for table in tables:
        cleaned = table.strip("`\"'()[]\n\t ")
        if cleaned:
            cleaned_tables.add(cleaned)

    return list(cleaned_tables)


def get_table_specific_info(full_info, target_tables):
    """Extract specific table information including schema and sample rows from the full schema."""
    all_blocks = full_info.split("CREATE TABLE")
    relevant_blocks = []

    for block in all_blocks[1:]:  #
        if any(f'"{table.upper()}"' in block.split("\n")[0] for table in target_tables):
            parts = block.split("/*")
            if len(parts) >= 2:
                schema = f"CREATE TABLE{parts[0]}"
                samples = f"/*{parts[1]}"
                relevant_blocks.append(f"{schema}{samples}")

    return "\n".join(relevant_blocks)


def post_process_sql(
    query,
    current_time="2105-12-31 23:59:00",
    precomputed_dict={
        "temperature": (35.5, 38.1),
        "sao2": (95.0, 100.0),
        "heart rate": (60.0, 100.0),
        "respiration": (12.0, 18.0),
        "systolic bp": (90.0, 120.0),
        "diastolic bp": (60.0, 90.0),
        "mean bp": (60.0, 110.0),
    },
):
    query = query.lower()
    if "current_time" in query:
        query = query.replace("current_time", f"'{current_time}'")
    if re.search("[ \n]+([a-zA-Z0-9_]+_lower)", query) and re.search(
        "[ \n]+([a-zA-Z0-9_]+_upper)", query
    ):
        vital_lower_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_lower)", query)[0]
        vital_upper_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_upper)", query)[0]
        vital_name_list = list(
            set(
                re.findall("([a-zA-Z0-9_]+)_lower", vital_lower_expr)
                + re.findall("([a-zA-Z0-9_]+)_upper", vital_upper_expr)
            )
        )
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace("_", " ")
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(
                    vital_upper_expr, f"{vital_range[1]}"
                )
    query = query.replace("''", "'").replace("< =", "<=")
    query = query.replace("%y", "%Y").replace("%j", "%J")
    query = query.replace("'now'", f"'{current_time}'")
    return query



def execute_wrapper(sql, timeout, db_path, tag, skip_indicator="null"):
    if sql != skip_indicator:
        try:
            result = func_timeout(timeout, execute_sqlite, args=(sql, db_path))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f"timeout_{tag}",)]
        except:
            result = [(f"error_{tag}",)]

        result = str(sorted([str(ret) for ret in result[:100]]))
    else:
        result = skip_indicator
    return result


def execute_query(sql1, sql2, timeout, db_path, data_idx=None):
    """
    Execute the query. Time out if it exceeds {args.timeout} seconds
    """
    result1 = execute_wrapper(sql1, timeout, db_path, tag="real")
    result2 = execute_wrapper(sql2, timeout, db_path, tag="pred")
    result = {"data_idx": data_idx, "real": result1, "pred": result2}
    return result


def evaluate_single_prediction(truth_query, pred_query, timeout, db_path):
    """Evaluate a single prediction"""
    ret = execute_query(truth_query, pred_query, timeout, db_path)
    return ret["real"], ret["pred"]



class cypher_evaluator(BaseModel):
    grade: Optional[str] = Field(default=None, description="The cypher evaluation")


parser_eval = PydanticOutputParser(pydantic_object=cypher_evaluator)


def contains_cypher_tags(input_string):
    pattern = r"```Cypher(.*?)```"
    match = re.search(pattern, input_string, re.IGNORECASE | re.DOTALL)

    return match is not None




def extract_xml_answer_cypher(text: str) -> str:
    text = text.lower()
    if "```cypher" in text:
        sql_parts = text.split("```cypher")[1:]

        if sql_parts:
            last_sql_part = sql_parts[-1]
            if "```" in last_sql_part:
                query = last_sql_part.split("```")[0]

                return query.strip().upper()
    print("No cypher query found in response")
    return " "


def extract_xml_answer_sql(text: str) -> str:
    """
    Extract the last SQL query from a response that contains ```sql and ``` markers.
    Falls back to xml extraction if SQL markers aren't found.
    """
    if "```sql" in text:
        sql_blocks = text.split("```sql")

        if len(sql_blocks) > 1:
            last_sql_part = sql_blocks[-1]
            if "```" in last_sql_part:
                query = last_sql_part.split("```")[0]
                return query.strip()
    print("No SQL query found in response")
    return " "

def extract_SM3_answer(text: str, query_language: str) -> str:
    """
    Extract query from LLM response based on the query language.
    
    Args:
        text (str): The LLM response text
        query_language (str): One of "MQL", "SQL", "Cypher", "SPARQL"
        
    Returns:
        str: The extracted query or empty string if no query found
    """
    text = text.lower()
    query_language = query_language.lower()
    
    language_markers = {
        "mql": "```mongodb",  # MongoDB uses this marker
        "sql": "```sql",
        "cypher": "```cypher",
        "sparql": "```sparql"
    }
    
    marker = language_markers.get(query_language)
    if not marker:
        print(f"Unsupported query language: {query_language}")
        return " "
    
    if marker in text:
        query_parts = text.split(marker)[1:]
        if query_parts:
            last_query_part = query_parts[-1]
            if "```" in last_query_part:
                query = last_query_part.split("```")[0]
                return query.strip().upper()
    
    print(f"No {query_language} query found in response: {text}")
    return " "



def evaluate_generated_response(llm: ChatOpenAI, sql_eval: List[str]):

    eval_prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser_eval.get_format_instructions()
        },
    )

    eval_chain = eval_prompt | llm | parser_eval
    eval_res = eval_chain.invoke(sql_eval)

    try:
        return 1 if eval_res.grade.lower() == "correct" else 0
    except AttributeError:
        return 0


def get_EM_score(gold_answer, llm_answer):
    def normalize(text):
        text = str(text).lower()
        punctuation = """!()-[]{};:'"\,<>./?@#$%^&*_~"""
        normalized_text = "".join(char for char in text if char not in punctuation)
        return " ".join(normalized_text.split()).strip()

    normalized_gold = normalize(gold_answer)
    normalized_llm = normalize(llm_answer)

    try:
        gold_float = float(normalized_gold)
        llm_float = float(normalized_llm)
        return 1 if round(gold_float, 2) == round(llm_float, 2) else 0
    except ValueError:
        return 1 if normalized_gold == normalized_llm else 0



def evaluate_rouge_response_manual(reference,prediction):
    def get_lcs_length(x, y):
        # Create LCS matrix
        m, n = len(x), len(y)
        lcs = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    lcs[i][j] = lcs[i-1][j-1] + 1
                else:
                    lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
        
        return lcs[m][n]

    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    
    lcs_length = get_lcs_length(pred_words, ref_words)
    
    if len(pred_words) == 0:
        precision = 0.0
    else:
        precision = lcs_length / len(pred_words)
        
    if len(ref_words) == 0:
        recall = 0.0
    else:
        recall = lcs_length / len(ref_words)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1



def evaluate_bleu_response_manual(true_answer: str, generated_answer: str, normalize: bool = False):
    """
    Evaluate BLEU score between a generated answer and a reference answer.
    """
    def normalize_answer(text):
        # Add your normalization logic here if needed
        return text
    
    def get_ngrams(text, n):
        """Get n-grams from text."""
        words = text.lower().strip().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i + n]))
        return ngrams
    
    def count_ngrams(ngrams):
        """Count occurrences of each n-gram."""
        counter = {}
        for ngram in ngrams:
            counter[ngram] = counter.get(ngram, 0) + 1
        return counter
    
    def modified_precision(candidate, reference, n):
        """Calculate modified precision for n-grams."""
        candidate_ngrams = get_ngrams(candidate, n)
        reference_ngrams = get_ngrams(reference, n)
        
        if not candidate_ngrams:
            return 0
        
        candidate_counts = count_ngrams(candidate_ngrams)
        reference_counts = count_ngrams(reference_ngrams)
        
        total_matches = 0
        for ngram, count in candidate_counts.items():
            total_matches += min(count, reference_counts.get(ngram, 0))
            
        return total_matches / len(candidate_ngrams) if candidate_ngrams else 0
    
    try:
        if normalize:
            true_answer = normalize_answer(true_answer)
            generated_answer = normalize_answer(generated_answer)
        
        true_answer = true_answer.lower().strip()
        generated_answer = generated_answer.lower().strip()
        
        precisions = []
        for n in range(1, 5):
            p = modified_precision(generated_answer, true_answer, n)
            precisions.append(p if p != 0 else 1e-7)  # Avoid log(0)
        
        ref_words = len(true_answer.split())
        gen_words = len(generated_answer.split())
        brevity_penalty = min(1, exp(1 - ref_words/gen_words)) if gen_words > 0 else 0
        
        log_precisions = sum(log(p) for p in precisions) / 4
        bleu = brevity_penalty * exp(log_precisions)
        
        scores = {
            'bleu': bleu,
            'precisions': precisions,
            'brevity_penalty': brevity_penalty,
            'length_ratio': gen_words / ref_words if ref_words > 0 else 0,
            'translation_length': gen_words,
            'reference_length': ref_words
        }
        
        return scores['bleu']
        
    except Exception as e:
        print(f"Error evaluating BLEU: {e}", 'gen:', generated_answer, 'true:', true_answer)
        return 0




def calculate_average_scores(results: List[Dict[str, Dict[str, float]]]) -> str:
    totals = defaultdict(int)
    counts = defaultdict(int)

    for res in results:
        if res:
            for score_name, score_value in res['scores'].items():
                totals[score_name] += score_value
                counts[score_name] += 1

    average_scores = {subject: round(totals[subject] / counts[subject], 3) 
                      for subject in totals if counts[subject] > 0}

    f1_scores = {}
    if "precision_ans" in average_scores and "recall_ans" in average_scores:
        avg_precision_ans = average_scores["precision_ans"]
        avg_recall_ans = average_scores["recall_ans"]
        f1_scores["f1_ans"] = round((2 * (avg_precision_ans * avg_recall_ans) /
                                      (avg_precision_ans + avg_recall_ans + 1e-10)), 3)

    if "precision_exec" in average_scores and "recall_exec" in average_scores:
        avg_precision_exec = average_scores["precision_exec"]
        avg_recall_exec = average_scores["recall_exec"]
        f1_scores["f1_exec"] = round((2 * (avg_precision_exec * avg_recall_exec) /
                                       (avg_precision_exec + avg_recall_exec + 1e-10)), 3)

    final_scores = {**average_scores, **f1_scores}
    
    final_string = ', '.join(f'{subject}: {avg}' for subject, avg in final_scores.items())
    return final_scores, final_string
