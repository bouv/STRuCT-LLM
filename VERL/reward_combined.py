import torch
import os
import numpy as np
import random
import wandb
import json
import difflib
import logging
import re
import networkx as nx
from nltk.util import ngrams
from typing import List, Set, Optional, Dict
from pydantic import BaseModel, Field
from datasets import DatasetDict, load_dataset, interleave_datasets
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import csv
import sqlite3
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.utilities import SQLDatabase
from sqlglot import exp, parse_one
  
np.random.seed(88)

# Replace with your model name as needed
llm = ChatOpenAI(
    model='openai_gpt41_nano',
    api_key="your_api_key_here",  # Anonymized
    base_url="https://api.your_base_url_here.com/v1",  # Anonymized
    model_kwargs={'temperature': 1}
)

################ helper functions ####################

def normalize_sql_or_cypher_query(query: str, query_type) -> str:
    if query_type == 'sql':
        try:
            # Parse the single SQL query to get the expression tree
            expression_tree = parse_one(query)

            # Define a transformer function to manipulate nodes
            def transformer(node):
                # Example transformation: change column names or functions
                if isinstance(node, exp.Column) and node.name == "a":
                    return parse_one("FUN(a)")
                return node

            # Apply the transformation to the expression tree
            transformed_tree = expression_tree.transform(transformer)

            # Generate the SQL from the transformed tree
            return transformed_tree.sql()

        except Exception as e:
            # Log the error if needed (optional)
            print(f"Error normalizing query: {e}")
            # Return the original query if an error occurs
            return query
    elif query_type == 'cypher':
        # Normalize Cypher query (e.g., uppercase)
        return query.upper()
    else:
        raise ValueError("Unsupported query type. Use 'sql' or 'cypher'.")


def extract_sql_or_cypher_answer(text: str, query_type: str) -> str:
    """
    Extract the last SQL query from a response that contains ```sql and ``` markers.
    Returns empty string if no SQL markers are found.
    """
    text=text.lower()
    tags="```cypher" if query_type == 'cypher' else "```sql"
    # Find all occurrences of SQL code blocks
    if tags in text:
        # Split at ```sql and remove the first part
        sql_parts = text.split(tags)[1:]
        
        if sql_parts:
            # Take the last sql part
            last_sql_part = sql_parts[-1]
            # Split at the next ``` and take everything before it
            if "```" in last_sql_part:
                query = last_sql_part.split("```")[0]
                
                return normalize_sql_or_cypher_query(query.strip(), query_type)
    
    # If no valid SQL block is found, return empty string
    print("No query found in response")
    return ""


def calculate_clause_similarity(pred_content: str, true_content: str) -> float:
    """Calculate similarity between the content following a keyword"""
    if not pred_content or not true_content:
        return 0.0
    
    # Normalize content
    pred_content = ' '.join(pred_content.lower().split())
    true_content = ' '.join(true_content.lower().split())
    
    # Use sequence matcher for content comparison
    return difflib.SequenceMatcher(None, pred_content, true_content).ratio()


############### STRING MATCHING ####################



def string_matching_reward(data_source, solution_str, ground_truth, extra_info=None):
    solution_str=extract_sql_or_cypher_answer(solution_str, data_source)
    ground_truth=normalize_sql_or_cypher_query(ground_truth, data_source)
    
    try:
        reward = difflib.SequenceMatcher(None, ground_truth, solution_str).ratio()
        print("\nPredicted Query:", solution_str)
        print("True Query:", ground_truth)
        print('string reward', reward)

        return reward
    
    except Exception as e:
        print(f"Error calculating string match: {e}")
        return 0



############### GED ####################

# ——— helper to build graphs with label attributes ———
def build_graph(cypher: str) -> nx.DiGraph:
    G = nx.DiGraph()
    
    # Node pattern (unchanged)
    node_pattern = r'\((\w+)(?::(\w+))?(?:\{[^}]*\})?\)'
    
    # Updated relationship pattern to handle property maps and unnamed relationships
    rel_pattern = r'\((\w+)(?::(\w+))?(?:\{[^}]*\})?\)-\[(\w+)?(?::(\w+))?(?:\{[^}]*\})?\]->\((\w+)(?::(\w+))?(?:\{[^}]*\})?\)'
    undirected_rel_pattern = r'\((\w+)(?::(\w+))?(?:\{[^}]*\})?\)-\[(\w+)?(?::(\w+))?(?:\{[^}]*\})?\]-\((\w+)(?::(\w+))?(?:\{[^}]*\})?\)' #NEW
    
    # Parse nodes
    for match in re.findall(node_pattern, cypher):
        var, label = match
        if label:  # Only add label if it exists
            G.add_node(var, label=label)
        else:
            G.add_node(var)
            
    # Parse relationships
    for match in re.findall(rel_pattern, cypher):
        src, src_label, rel_var, rel_type, tgt, tgt_label = match
        G.add_edge(src, tgt, type=rel_type if rel_type else None)
    
    for match in re.findall(undirected_rel_pattern, cypher): #NEW
        src, src_label, rel_var, rel_type, tgt, tgt_label = match
        # Add both directions for undirected edges
        G.add_edge(src, tgt, type=rel_type if rel_type else None, directed=False)
        G.add_edge(tgt, src, type=rel_type if rel_type else None, directed=False)

    print("\nNodes and attributes:")
    for node in G.nodes(data=True):
        print(f"Node: {node[0]}, Attributes: {node[1]}")
    
    print("\nEdges and attributes:")
    for edge in G.edges(data=True):
        print(f"Edge: {edge[0]} -> {edge[1]}, Attributes: {edge[2]}")
        
    return G

# ——— custom cost functions ———
def node_subst_cost(n1_attrs, n2_attrs):
    # zero cost if same label, else cost=1
    #print('node', 0 if n1_attrs.get("label") == n2_attrs.get("label") else 1)
    return 0 if n1_attrs.get("label") == n2_attrs.get("label") else 1

def edge_subst_cost(e1_attrs, e2_attrs):
    # zero cost if same relationship type, else cost=1
    #print('edge,', 0 if e1_attrs.get("type") == e2_attrs.get("type") else 1)
    return 0 if e1_attrs.get("type") == e2_attrs.get("type") else 1

# (you can leave deletion/insertion at their default cost=1)

# ——— compute normalized GED overlap ———
def ged_overlap_with_labels(cypher1: str, cypher2: str) -> float:
    G1 = build_graph(cypher1)
    G2 = build_graph(cypher2)
    ged = nx.graph_edit_distance(
        G1,
        G2,
        node_subst_cost=node_subst_cost,
        edge_subst_cost=edge_subst_cost
    )
    print(ged)
    size1 = G1.number_of_nodes() + G1.number_of_edges()
    size2 = G2.number_of_nodes() + G2.number_of_edges()
    return 1 - ged / max(size1, size2)


def cypher_ged_reward(data_source, solution_str, ground_truth, extra_info=None):
    solution_str=extract_sql_or_cypher_answer(solution_str, data_source)
    ground_truth=normalize_sql_or_cypher_query(ground_truth, data_source)
    
    try:
        reward = ged_overlap_with_labels( ground_truth, solution_str)
        print("\nPredicted Query:", solution_str)
        print("True Query:", ground_truth)
        print('ged reward', reward)

        return reward
    
    except Exception as e:
        print(f"Error calculating ged: {e}")
        return 0

############### COMPONENT MATCHING ####################


class CypherSimilarityChecker:
    def __init__(self):
        self.keywords = {
            "main_body": ["MATCH", "RETURN", "WHERE", "AND", "OR", "NOT"],
            "patterns": ["()-", "[:]", "->", "<-"],
            "clause": ["ORDER BY", "LIMIT", "SKIP"],
            "functions": ["COUNT", "SUM", "AVG", "MIN", "MAX"],
            "operators": ["=", ">", "<", ">=", "<=", "!="],
        }

    def _extract_component_after_keyword(self, cypher_query, pos, keyword):
        """Extract meaningful component after keyword."""
        # Find where to stop extracting
        end_pos = len(cypher_query)
        # Find the next keyword position
        for kw in self.keywords["main_body"]:
            next_kw_pos = cypher_query.find(kw, pos)
            if next_kw_pos != -1 and next_kw_pos < end_pos:
                end_pos = next_kw_pos
        # Return the cleaned component
        component = cypher_query[pos+2:end_pos].strip()
        # Remove unwanted characters (H, N, etc.)
        return component

    def parse_cypher(self, cypher_query):
        """Parse Cypher query into components by keyword category."""
        components = defaultdict(list)
        cypher_query = f" {cypher_query} "  # Add spacing for easier parsing
        cypher_upper = cypher_query.upper()
        
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                spaced_keyword = f" {keyword} "
                start_pos = 0
                while True:
                    pos = cypher_upper.find(spaced_keyword, start_pos)
                    if pos == -1:
                        break
                    actual_pos = pos + len(keyword)  # Position after the keyword
                    # Extract the meaningful component
                    component = self._extract_component_after_keyword(cypher_query, actual_pos, keyword)
                    if component:
                        components[category].append((keyword, component))
                    start_pos = pos + len(keyword)
        
        return components

    def _calculate_f1(self, list1, list2):
        """Calculate F1 score between two lists of components."""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        precision = intersection / len(set1)
        recall = intersection / len(set2)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _normalize_component(self, component):
        """Normalize the component by replacing variable names with 'VAR' while keeping the labels."""
        
        # Match nodes in the form (z:...) and replace only the variable name before ':'
        # Capture the variable name and the label separately
        matches = re.findall(r'\((\s*\w+)\s*:(\w+)\s*\)', component)
        # Replace each variable name before ':' with VAR and store it for later replacement
        for match in matches:
            variable_name = match[0].strip()  # Extract the variable part
            #print(variable_name)
            component = component.replace(variable_name, 'VAR')  # Replace all occurrences of the variable
    
        # Now replace the variable in the node with VAR:label format
        component = re.sub(r'\(\s*\w+\s*:(\w+)\s*\)', r'(VAR:\1)', component)
    
        return component

    def calculate_similarity(self, cypher1, cypher2):
        """Calculate similarity between two Cypher queries."""
        components1 = self.parse_cypher(self._normalize_component(cypher1))
        components2 = self.parse_cypher(self._normalize_component(cypher2))
        #print(components1)
        #print(components2)
    
        # Flatten all components into single lists
        all_components1 = [[comp[0],comp[1]] for category_components in components1.values() for comp in category_components]
        all_components2 = [[comp[0],comp[1]] for category_components in components2.values() for comp in category_components]
        #print(all_components1)
        #print(all_components2)

    
        # Calculate F1 score for each keyword
        keyword_scores = {}
        all_keywords = set([comp[0] for comp in components1['main_body']] + [comp[0] for comp in components2['main_body']])
        #print(all_keywords)
    
        for keyword in all_keywords:
            components_1 = [comp[1] for comp in all_components1 if len(comp) > 1 and comp[0] == keyword]
            #print(components_1)
            components_2 = [comp[1] for comp in all_components2 if len(comp) > 1 and comp[0] == keyword]
            #print(components_2)
    
            if components_1 or components_2:  # Only calculate F1 if at least one of the lists is non-empty
                score = self._calculate_f1(components_1, components_2)
                keyword_scores[keyword] = score
        print(keyword_scores)
    
        # Calculate average of all keyword scores
        final_score = sum(keyword_scores.values()) / len(keyword_scores) if keyword_scores else 0.0
        return final_score





class SQLSimilarityChecker:
    def __init__(self):
        self.keywords = {
            "main_body": ["SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", 
                         "EXISTS", "IS", "NULL", "IIF", "CASE", "CASE WHEN"],
            "join": ["INNER JOIN", "LEFT JOIN", "ON", "AS"],
            "clause": ["BETWEEN", "LIKE", "LIMIT", "ORDER BY", "ASC", "DESC", 
                      "GROUP BY", "HAVING", "UNION", "ALL", "EXCEPT", "PARTITION BY"],
            "aggregation": ["AVG", "COUNT", "MAX", "MIN", "ROUND", "SUM"],
            "scalar": ["ABS", "LENGTH", "STRFTIME", "JULIADAY", "NOW", "CAST", 
                      "SUBSTR", "INSTR"],
            #"comparison": ["=", ">", "<", ">=", "<=", "!="],
            #"computing": ["-", "+", "*", "/"]
        }
        
    def _extract_comma_separated(self, sql, start_pos):
        """Extract comma-separated items until next keyword"""
        next_keyword_pos = len(sql)
        sql_upper = sql.upper()
        for category_keywords in self.keywords.values():
            for keyword in category_keywords:
                pos = sql_upper.find(' '+keyword+' ', start_pos)
                if pos != -1 and pos < next_keyword_pos:
                    next_keyword_pos = pos
        
        content = sql[start_pos:next_keyword_pos].strip()
        items = [item.strip() for item in content.split(',')]
        return tuple(filter(None, items))  # Return tuple instead of set
        
    def _extract_comparison(self, sql, pos):
        """Extract comparison expression"""
        next_pos = len(sql)
        sql_upper = sql.upper()
        for category_keywords in self.keywords.values():
            for keyword in category_keywords:
                keyword_pos = sql_upper.find(keyword, pos + 1)
                if keyword_pos != -1 and keyword_pos < next_pos:
                    next_pos = keyword_pos
        
        return sql[pos:next_pos].strip()
        
    def _extract_case_statement(self, sql, pos):
        """Extract complete CASE statement"""
        sql_upper = sql.upper()
        end_pos = sql_upper.find("END", pos)
        if end_pos != -1:
            return sql[pos:end_pos + 3].strip()
        return ""
        
    def _extract_component(self, sql, pos, keyword):
        """Extract meaningful component after keyword"""
        keyword_upper = keyword.upper()
        
        # Find the next keyword with proper spacing
        def find_next_keyword_pos(start_pos):
            next_pos = len(sql)
            for cat_keywords in self.keywords.values():
                for k in cat_keywords:
                    # Add spaces around keyword for exact matching
                    spaced_k = f" {k} "
                    k_pos = sql.upper().find(spaced_k, start_pos)
                    if k_pos != -1 and k_pos < next_pos:
                        next_pos = k_pos
                    # Check for keywords followed by special characters
                    for special_char in ['(', ')', ',', '\n']:
                        spaced_k_special = f" {k}{special_char}"
                        k_pos = sql.upper().find(spaced_k_special, start_pos)
                        if k_pos != -1 and k_pos < next_pos:
                            next_pos = k_pos
            return next_pos
    
        if keyword_upper in ["SELECT", "GROUP BY", "ORDER BY", "WHERE"]:
            return self._extract_comma_separated(sql, pos + len(keyword))
            
        elif keyword_upper in ["CASE", "CASE WHEN"]:
            return self._extract_case_statement(sql, pos)
            
        #elif keyword_upper in self.keywords["comparison"]:
        #    return self._extract_comparison(sql, pos)
            
        else:
            next_pos = find_next_keyword_pos(pos + len(keyword))
            return sql[pos + len(keyword):next_pos].strip()

    def parse_sql(self, sql_query):
        """Parse SQL query into components by keyword category"""
        components = defaultdict(list)
        
        # Add spaces around the query to help with boundary detection
        sql_query = f" {sql_query} "
        sql_upper = sql_query.upper()
        
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                # Add spaces around keyword for exact matching
                spaced_keyword = f" {keyword} "
                start_pos = 0
                
                while True:
                    pos = sql_upper.find(spaced_keyword, start_pos)
                    if pos == -1:
                        # Try with trailing special characters
                        for special_char in ['(', ')', ',', '\n']:
                            spaced_keyword_special = f" {keyword}{special_char}"
                            pos = sql_upper.find(spaced_keyword_special, start_pos)
                            if pos != -1:
                                break
                        if pos == -1:
                            break
                    
                    # Extract the component following the keyword
                    # Add 1 to skip the leading space we added
                    actual_pos = pos + 1
                    component = self._extract_component(sql_query, actual_pos, keyword)
                    if component:
                        components[category].append((keyword, component))
                    
                    start_pos = pos + len(keyword)
        
        return components

    def _calculate_f1(self, list1, list2):
        """Calculate F1 score between two lists of components"""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
            
        # Convert to sets of strings for comparison
        set1 = {str(item) for item in list1}
        set2 = {str(item) for item in list2}
            
        intersection = len(set1.intersection(set2))
        precision = intersection / len(set1)
        recall = intersection / len(set2)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)

    def calculate_similarity(self, sql1, sql2):
        """Calculate similarity between two SQL queries"""
        # Replace AND/OR with commas
        sql1 = sql1.replace(" AND ", " , ").replace(" OR ", " , ")
        sql2 = sql2.replace(" AND ", " , ").replace(" OR ", " , ")
        sql1 = sql1.replace(" and ", " , ").replace(" or ", " , ")
        sql2 = sql2.replace(" and ", " , ").replace(" or ", " , ")
        
        # Get all components without caring about categories
        components1 = self.parse_sql(sql1)
        components2 = self.parse_sql(sql2)
        
        # Flatten all components into single lists
        all_components1 = []
        all_components2 = []
        
        for category_components in components1.values():
            all_components1.extend(category_components)
            
        for category_components in components2.values():
            all_components2.extend(category_components)

        print(all_components1)
        # Calculate F1 score for each keyword
        keyword_scores = {}
        all_keywords = set([comp[0] for comp in all_components1] + [comp[0] for comp in all_components2])
        
        for keyword in all_keywords:
            # Get components for this keyword from both queries
            components_1 = [comp[1] for comp in all_components1 if comp[0] == keyword ]
            components_2 = [comp[1] for comp in all_components2 if comp[0] == keyword]
            print(f"\nKeyword: {keyword}")
            print(f"Query 1 components: {components_1}")
            print(f"Query 2 components: {components_2}")
            
            # Calculate F1 score for this keyword
            score = self._calculate_f1(components_1[0] if components_1 !=[] else components_1, components_2[0] if components_2 !=[] else components_2)
            keyword_scores[keyword] = score
            
            # Print components and score for this keyword

            print(f"F1 Score: {score}")
        
        # Calculate average of all keyword scores
        final_score = sum(keyword_scores.values()) / len(keyword_scores) if keyword_scores else 0.0
        print(f"\nFinal Similarity Score (average of all keyword F1 scores): {final_score}")
        return final_score





def component_matching_new(data_source, solution_str, ground_truth, extra_info=None):
    solution_str=extract_sql_or_cypher_answer(solution_str, data_source)
    ground_truth=normalize_sql_or_cypher_query(ground_truth, data_source)

    checker = SQLSimilarityChecker() if data_source == 'sql' else CypherSimilarityChecker()
    
    try:
        reward = checker.calculate_similarity(solution_str, ground_truth)
        print("\nPredicted Query:", solution_str)
        print("True Query:", ground_truth)
        print('component_matching_new reward', reward)

        return reward
    
    except Exception as e:
        print(f"Error calculating component_matching_new: {e}")
        return 0


###############################################



class Components_class(BaseModel):
    keywords_content: Dict[str, str]


def component_matching_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function analyzing keywords and their following content
    """

    
    parser = PydanticOutputParser(pydantic_object=Components_class)
    solution_str=extract_sql_or_cypher_answer(solution_str, data_source)
    ground_truth=normalize_sql_or_cypher_query(ground_truth, data_source)
    
    extract_template_cypher = """You are a Cypher query expert. Analyze this Cypher query and break it down by keywords and their following content.
For each Cypher keyword (MATCH, WHERE, RETURN, etc.), extract:
1. The keyword itself (lowercase)
2. The content that follows it until the next keyword
IMPORTANT PATTERN HANDLING:
- Combine node and relationship patterns under 'match'
- Include complete path patterns and their variables
- Example: For "MATCH (p:Person)-[:KNOWS]->(f:Person)"
  Use: "match": "(p:Person)-[:KNOWS]->(f:Person)"
Example format:
{{
    "match": "(p:Person)-[:WORKS_AT]->(c:Company)",
    "where": "p.age > 25",
    "with": "p, c.name as company",
    "return": "p.name, company",
    "order by": "p.name desc",
    "limit": "10"
}}
Rules:
1. Always include complete path patterns
2. Normalize content (standard spacing, lowercase)
3. Handle multiple MATCH clauses as separate entries: "match1", "match2", etc.
4. Include any WITH clauses as separate components
5. Capture any CREATE, DELETE, or SET operations
Cypher Query: {query}
{format_instructions}
Return the keyword-content pairs in the specified JSON format, DONT CORRECT THE QUERY IF IT'S WRONG WE WANT TO SEE."""

    extract_template_sql = """You are an SQL expert. Analyze this SQL query and break it down by keywords and their following content.

For each SQL keyword (SELECT, FROM, WHERE, etc.), extract:
1. The keyword itself (lowercase)
2. The content that follows it until the next keyword

IMPORTANT JOIN HANDLING:
- Combine 'JOIN ... ON' as a single component under 'join'
- Include the complete join condition (both the table and the ON clause)
- Example: For "JOIN departments d ON e.department_id = d.id"
  Use: "join": "departments d on e.department_id = d.id"

Example format:
{{
    "select": "id, name, age",
    "from": "users",
    "join": "departments d on user_id = d.id",
    "where": "age > 25",
    "group by": "department_name",
    "order by": "name desc"
}}

Rules:
1. Always combine JOIN and ON clauses together
2. Normalize content (standard spacing, lowercase)
3. Treat all types of joins (INNER JOIN, LEFT JOIN, etc.) the same way
4. Multiple joins should be separate entries: "join1", "join2", etc.

SQL Query: {query}

{format_instructions}

Return the keyword-content pairs in the specified JSON format, DONT CORRECT THE QUERY IF IT'S WRONG WE WANT TO SEE."""


    

    eval_prompt = PromptTemplate(
    template=extract_template_sql if data_source == 'sql' else extract_template_cypher,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    try:
        # Extract keyword-content pairs for both queries using LLM
        eval_chain = eval_prompt | llm | parser
        
        pred_components = eval_chain.invoke({"query": solution_str}).keywords_content
        true_components = eval_chain.invoke({"query": ground_truth}).keywords_content
        
        # Get all unique keywords from both queries
        all_keywords = set(pred_components.keys()).union(set(true_components.keys()))
        
        # Calculate scores for each keyword
        keyword_scores = {}
        for keyword in all_keywords:
            if keyword not in pred_components:
                # Keyword missing from prediction
                keyword_scores[keyword] = 0.0
            elif keyword not in true_components:
                # Extra keyword in prediction
                keyword_scores[keyword] = 0.0
            else:
                # Keyword present in both, compare content
                keyword_scores[keyword] = calculate_clause_similarity(
                    pred_components[keyword],
                    true_components[keyword]
                )
        
        # Calculate final score as average of keyword scores
        final_score = sum(keyword_scores.values()) / len(keyword_scores)
        
        # Detailed logging
        print(f"\nPredicted Query: {solution_str}")
        print(f"True Query: {ground_truth}")
        print(f"Predicted Components: {pred_components}")
        print(f"True Components: {true_components}")
        print(f"Keyword Scores:")
        for keyword, score in keyword_scores.items():
            print(f"  {keyword}: {score}")
            if score == 0.0:
                if keyword not in pred_components:
                    print(f"    Missing in prediction")
                elif keyword not in true_components:
                    print(f"    Extra in prediction")
        print(f"llm component reward: {final_score}")
        
        return final_score
            
    except Exception as e:
        print(f"Error in component matching: {e}")
        return 0
    









##################



class QueryEvaluation(BaseModel):
    grade: str = Field(
        description="grade for the query: 'Very bad', 'Bad', 'Average', 'Above average', 'Good', or 'Excellent'"    )

def llm_scoring_classes_judge_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function that uses LLM to grade SQL queries with categorical ratings.
    Returns list of rewards based on grades.
    """

    solution_str=extract_sql_or_cypher_answer(solution_str, data_source)
    ground_truth=normalize_sql_or_cypher_query(ground_truth, data_source)

    
    # Grade to reward mapping
    grade_rewards = {
        'Excellent': 1.0,
        'Good': 0.8,
        'Above average': 0.6,
        'Bad': 0.2,
        'Very bad': 0.0
    }
    
    parser = PydanticOutputParser(pydantic_object=QueryEvaluation)
    
    eval_template_cypher = """Compare the Cypher query to the correct query and grade it as: 'Very bad', 'Bad', 'Above average', 'Good', or 'Excellent'.
    This is the following grading system, use the correct query as reference:
    
    - Correct Query: {ground_truth}
    
    1. Excellent: Only given when the Cypher query is perfect and matches {ground_truth}, including:
       - Correct node labels and relationship types
       - Proper pattern matching syntax
       - Correct variable naming and property access
       - Exact match of all conditions and return values

    2. Good: When there are minor issues such as:
       - Small syntax variations (e.g., different variable names but correct pattern)
       - Equivalent but differently structured patterns
       - Minor differences in property access syntax
       - Different but equivalent WHERE conditions

    3. Above average: When the query has the right idea but contains:
       - Incorrect relationship direction
       - Missing or extra node labels
       - Incomplete pattern matching
       - Partially incorrect property conditions
       - Different but related return values

    4. Bad: When the query has multiple issues:
       - Wrong pattern structure
       - Missing crucial relationships or nodes
       - Incorrect or missing WHERE clauses
       - Wrong aggregations or collections
       - Incorrect use of Cypher-specific features (WITH, UNWIND, etc.)

    5. Very bad: When the query:
       - Is not a valid Cypher query
       - Uses completely wrong patterns
       - Returns entirely different data
       - Contains fundamental misunderstandings of graph patterns
       - Is missing or has severe syntax errors

    Query to grade:
    {solution_str}
    {format_instructions}
    Return ONLY the JSON with grades list."""


    eval_template_sql = """Compare the SQL query to the correct query and grade it as: 'Very bad', 'Bad', 'Above average', 'Good', or 'Excellent'.
    This is the following grading system, use the correct query as reference :
    
    - Correct Query: {ground_truth}
    
    1. Excellent: this is only given when the SQL query is perfect and matches {ground_truth}
    2. Good: This is when there is a grammar mistake in the query
    3. Above average: This is when the query is mostly correct but gets a logical step wrong in the query
    4. Bad: Makes more than one mistake in the query 
    5. Very bad: does not produce a query or varies significantly from the correct query



    Query to grade:
    {solution_str}

    {format_instructions}

    Return ONLY the JSON with grades list."""

    
    try:  
        eval_prompt = PromptTemplate(
            template=eval_template_sql if data_source == 'sql' else eval_template_cypher,
            input_variables=["solution_str", "ground_truth"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        eval_chain = eval_prompt | llm | parser
        
        # Get grades for all queries
        eval_result = eval_chain.invoke({
            "solution_str": solution_str,
            "ground_truth": ground_truth,
        })
        print(eval_template_sql if data_source == 'sql' else eval_template_cypher)
        print(eval_result)

        reward= grade_rewards[eval_result.grade]

        print("\nPredicted Query:", solution_str)
        print("True Query:", ground_truth)
        print("classes Reward:", reward)

        
        # Convert grades to rewards
        return reward
            
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return 0    



def comp_ged_reward(data_source, solution_str, ground_truth, extra_info=None):

    try:
        # Calculate string similarity using SequenceMatcher
        if data_source == 'sql':
            reward=component_matching_new(data_source, solution_str, ground_truth)
        else:
            reward=ged_overlap_with_labels(extract_sql_or_cypher_answer(solution_str, data_source), normalize_sql_or_cypher_query(ground_truth, data_source))
        
        
        # Optional: Log the similarity and reward for tracking
        print("\nPredicted Query:", solution_str)
        print("True Query:", ground_truth)
        print("comp or ged matching Reward:", reward)

        return reward
    
    except Exception as e:
        print(f"Error calculating reward: {e}")
        return 0

def string_comp_classes_reward(data_source, solution_str, ground_truth, extra_info=None):
    return (component_matching_new(data_source, solution_str, ground_truth, extra_info=None)+llm_scoring_classes_judge_reward(data_source, solution_str, ground_truth, extra_info=None)+string_matching_reward(data_source, solution_str, ground_truth, extra_info=None))/3


def string_comp_ged_reward(data_source, solution_str, ground_truth, extra_info=None):
    return (comp_ged_reward(data_source, solution_str, ground_truth, extra_info=None)+llm_scoring_classes_judge_reward(data_source, solution_str, ground_truth, extra_info=None)+string_matching_reward(data_source, solution_str, ground_truth, extra_info=None))/3