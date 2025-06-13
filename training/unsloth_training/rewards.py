import os
import re
import numpy as np
import difflib
import networkx as nx
from pydantic import BaseModel, Field
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from sqlglot import exp, parse_one


np.random.seed(88)

llm = ChatOpenAI(model='openai_o3_mini',
                        api_key = os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_URL"),
                        model_kwargs={'temperature':1 }
                        )

################ helper functions ####################

def normalize_sql_or_cypher_query(query: str, query_type) -> str:
    if query_type == 'sql':
        try:
            expression_tree = parse_one(query)

            def transformer(node):
                if isinstance(node, exp.Column) and node.name == "a":
                    return parse_one("FUN(a)")
                return node

            transformed_tree = expression_tree.transform(transformer)

            return transformed_tree.sql()

        except Exception as e:
            return query
    elif query_type == 'cypher':
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
    if tags in text:
        sql_parts = text.split(tags)[1:]
        
        if sql_parts:
            last_sql_part = sql_parts[-1]
            if "```" in last_sql_part:
                query = last_sql_part.split("```")[0]
                
                return normalize_sql_or_cypher_query(query.strip(), query_type)
    
    print("No query found in response")
    return ""


def calculate_clause_similarity(pred_content: str, true_content: str) -> float:
    """Calculate similarity between the content following a keyword"""
    if not pred_content or not true_content:
        return 0.0
    
    pred_content = ' '.join(pred_content.lower().split())
    true_content = ' '.join(true_content.lower().split())
    
    return difflib.SequenceMatcher(None, pred_content, true_content).ratio()


############### STRING MATCHING ####################



def string_matching_reward(prompts, completions, answer, query_type, **kwargs) -> list[float]:
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_sql_or_cypher_answer(r, q) for r, q in zip(responses, query_type)]
    rewards = []
    
    for pred_query, true_query, single_query_type in zip(extracted_responses, answer, query_type):
        try:
            reward = difflib.SequenceMatcher(None, pred_query, normalize_sql_or_cypher_query(true_query, single_query_type)).ratio()
            
            
            print("\nPredicted Query:", pred_query)
            print("True Query:", true_query)
            print("string matching Reward:", reward)

            rewards.append(reward)
        
        except Exception as e:
            print(f"Error calculating string match: {e}")
            rewards.append(0.0)

    return rewards



############### GED ####################

# ——— helper to build graphs with label attributes ———
def build_graph(cypher: str) -> nx.DiGraph:
    G = nx.DiGraph()
    
    node_pattern = r'\((\w+)(?::(\w+))?(?:\{[^}]*\})?\)'
    
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
        
    return G

def node_subst_cost(n1_attrs, n2_attrs):
    return 0 if n1_attrs.get("label") == n2_attrs.get("label") else 1

def edge_subst_cost(e1_attrs, e2_attrs):
    return 0 if e1_attrs.get("type") == e2_attrs.get("type") else 1


def ged_overlap_with_labels(cypher1: str, cypher2: str) -> float:
    G1 = build_graph(cypher1)
    G2 = build_graph(cypher2)
    ged = nx.graph_edit_distance(
        G1,
        G2,
        node_subst_cost=node_subst_cost,
        edge_subst_cost=edge_subst_cost
    )
    size1 = G1.number_of_nodes() + G1.number_of_edges()
    size2 = G2.number_of_nodes() + G2.number_of_edges()
    return 1 - ged / max(size1, size2)

def cypher_ged_reward(prompts, completions, answer, **kwargs) -> list:
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [r for r in responses]
    rewards = []
    
    for pred_answer, true_query in zip(extracted_responses, answer):
        try:
            pred_query=extract_sql_or_cypher_answer(pred_answer, 'cypher')
            reward = ged_overlap_with_labels(normalize_sql_or_cypher_query(pred_query, 'cypher'), normalize_sql_or_cypher_query(true_query, 'cypher'))
            
            print("\nPredicted Query:", pred_query)
            print("True Query:", true_query)
            print("ged Reward:", reward)
            
            rewards.append(reward)
        
        except Exception as e:
            print(f"Error calculating ged: {e}")
            rewards.append(0.0)

    return rewards



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
        end_pos = len(cypher_query)
        for kw in self.keywords["main_body"]:
            next_kw_pos = cypher_query.find(kw, pos)
            if next_kw_pos != -1 and next_kw_pos < end_pos:
                end_pos = next_kw_pos
        component = cypher_query[pos+2:end_pos].strip()
        return component

    def parse_cypher(self, cypher_query):
        """Parse Cypher query into components by keyword category."""
        components = defaultdict(list)
        cypher_query = f" {cypher_query} " 
        cypher_upper = cypher_query.upper()
        
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                spaced_keyword = f" {keyword} "
                start_pos = 0
                while True:
                    pos = cypher_upper.find(spaced_keyword, start_pos)
                    if pos == -1:
                        break
                    actual_pos = pos + len(keyword)  
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

        matches = re.findall(r'\((\s*\w+)\s*:(\w+)\s*\)', component)
        for match in matches:
            variable_name = match[0].strip()  
            component = component.replace(variable_name, 'VAR')  
    
        component = re.sub(r'\(\s*\w+\s*:(\w+)\s*\)', r'(VAR:\1)', component)
    
        return component

    def calculate_similarity(self, cypher1, cypher2):
        """Calculate similarity between two Cypher queries."""
        components1 = self.parse_cypher(self._normalize_component(cypher1))
        components2 = self.parse_cypher(self._normalize_component(cypher2))

    
        all_components1 = [[comp[0],comp[1]] for category_components in components1.values() for comp in category_components]
        all_components2 = [[comp[0],comp[1]] for category_components in components2.values() for comp in category_components]

    
        keyword_scores = {}
        all_keywords = set([comp[0] for comp in components1['main_body']] + [comp[0] for comp in components2['main_body']])
    
        for keyword in all_keywords:
            components_1 = [comp[1] for comp in all_components1 if len(comp) > 1 and comp[0] == keyword]
            components_2 = [comp[1] for comp in all_components2 if len(comp) > 1 and comp[0] == keyword]
    
            if components_1 or components_2: 
                score = self._calculate_f1(components_1, components_2)
                keyword_scores[keyword] = score
    
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
        return tuple(filter(None, items)) 
        
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
        
        def find_next_keyword_pos(start_pos):
            next_pos = len(sql)
            for cat_keywords in self.keywords.values():
                for k in cat_keywords:
                    spaced_k = f" {k} "
                    k_pos = sql.upper().find(spaced_k, start_pos)
                    if k_pos != -1 and k_pos < next_pos:
                        next_pos = k_pos
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
 
        else:
            next_pos = find_next_keyword_pos(pos + len(keyword))
            return sql[pos + len(keyword):next_pos].strip()

    def parse_sql(self, sql_query):
        """Parse SQL query into components by keyword category"""
        components = defaultdict(list)
        
        sql_query = f" {sql_query} "
        sql_upper = sql_query.upper()
        
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                spaced_keyword = f" {keyword} "
                start_pos = 0
                
                while True:
                    pos = sql_upper.find(spaced_keyword, start_pos)
                    if pos == -1:
                        for special_char in ['(', ')', ',', '\n']:
                            spaced_keyword_special = f" {keyword}{special_char}"
                            pos = sql_upper.find(spaced_keyword_special, start_pos)
                            if pos != -1:
                                break
                        if pos == -1:
                            break

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
        sql1 = sql1.replace(" AND ", " , ").replace(" OR ", " , ")
        sql2 = sql2.replace(" AND ", " , ").replace(" OR ", " , ")
        sql1 = sql1.replace(" and ", " , ").replace(" or ", " , ")
        sql2 = sql2.replace(" and ", " , ").replace(" or ", " , ")
        
        components1 = self.parse_sql(sql1)
        components2 = self.parse_sql(sql2)
        
        all_components1 = []
        all_components2 = []
        
        for category_components in components1.values():
            all_components1.extend(category_components)
            
        for category_components in components2.values():
            all_components2.extend(category_components)

        keyword_scores = {}
        all_keywords = set([comp[0] for comp in all_components1] + [comp[0] for comp in all_components2])
        
        for keyword in all_keywords:
            components_1 = [comp[1] for comp in all_components1 if comp[0] == keyword ]
            components_2 = [comp[1] for comp in all_components2 if comp[0] == keyword]
            
            score = self._calculate_f1(components_1[0] if components_1 !=[] else components_1, components_2[0] if components_2 !=[] else components_2)
            keyword_scores[keyword] = score

        
        final_score = sum(keyword_scores.values()) / len(keyword_scores) if keyword_scores else 0.0
        return final_score

def component_matching_new(prompts, completions, answer, query_type, **kwargs) -> list[float]:
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_sql_or_cypher_answer(r, q) for r, q in zip(responses, query_type)]
    rewards = []
    
    for pred_query, true_query, single_query_type in zip(extracted_responses, answer, query_type):
        checker = SQLSimilarityChecker() if single_query_type == 'sql' else CypherSimilarityChecker()

        true_query=normalize_sql_or_cypher_query(true_query, single_query_type)
        try:
            similarity = checker.calculate_similarity(pred_query, true_query)
            
            reward = similarity  
            
            print("\nPredicted Query:", pred_query)
            print("True Query:", true_query)
            print("comp matching reward:", similarity)

            rewards.append(reward)
        
        except Exception as e:
            print(f"Error calculating string match: {e}")
            rewards.append(0.0)

    return rewards



def comp_ged_reward(prompts, completions, answer, query_type, **kwargs) -> list[float]:
    if query_type[0] == 'sql':
        try:
            return component_matching_new(prompts, completions, answer, **kwargs)
        except Exception as e:
            print(f"Error in component matching: {e}")
            return [0.0] * len(completions)
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_sql_or_cypher_answer(r, q) for r, q in zip(responses, query_type)]
    rewards = []
    
    for pred_query, true_query, single_query_type in zip(extracted_responses, answer, query_type):
        try:
            if single_query_type != 'sql':
                reward=ged_overlap_with_labels(normalize_sql_or_cypher_query(pred_query,single_query_type), normalize_sql_or_cypher_query(true_query,single_query_type))
            
            
            print("\nPredicted Query:", pred_query)
            print("True Query:", true_query)
            print("comp or ged matching Reward:", reward)

            rewards.append(reward)
        
        except Exception as e:
            print(f"Error calculating reward: {e}")
            rewards.append(0.0)

    return rewards





class QueryEvaluation(BaseModel):
    grades: list[str] = Field(
        description="List of grades for each query: 'Very bad', 'Bad', 'Average', 'Above average', 'Good', or 'Excellent'",
        min_items=1
    )

def llm_scoring_classes_judge_reward(prompts, completions, answer, query_type, **kwargs) -> list[float]:
    """
    Reward function that uses LLM to grade SQL queries with categorical ratings.
    Returns list of rewards based on grades.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_sql_or_cypher_answer(r, q) for r, q in zip(responses, query_type)]
    prompt_keys = [str(prompt) for prompt in prompts]
    
    grade_rewards = {
        'Excellent': 1.0,
        'Good': 0.8,
        'Above average': 0.6,
        'Bad': 0.2,
        'Very bad': 0.0
    }
    
    query_groups = {}
    for i, (prompt_key, pred_query, true_query, single_query_type) in enumerate(zip(prompt_keys, extracted_responses, answer, query_type)):
        true_query=normalize_sql_or_cypher_query(true_query, single_query_type)
        if prompt_key not in query_groups:
            query_groups[prompt_key] = []
        query_groups[prompt_key].append({
            'index': i,
            'pred_query': pred_query,
            'true_query': true_query,
            'single_query_type': single_query_type
        })
    
    parser = PydanticOutputParser(pydantic_object=QueryEvaluation)
    
    eval_template_cypher = """Compare these Cypher queries to the correct query and grade each one as: 'Very bad', 'Bad', 'Above average', 'Good', or 'Excellent'.
    This is the following grading system, use the correct query as reference:
    
    - Correct Query: {true_query}
    
    1. Excellent: Only given when the Cypher query is perfect and matches {true_query}, including:
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

    Queries to grade:
    {queries_to_rank}
    {format_instructions}
    Return ONLY the JSON with grades list."""


    eval_template_sql = """Compare these SQL queries to the correct query and grade each one as: 'Very bad', 'Bad', 'Above average', 'Good', or 'Excellent'.
    This is the following grading system, use the correct query as reference :
    
    - Correct Query: {true_query}
    
    1. Excellent: this is only given when the SQL query is perfect and matches {true_query}
    2. Good: This is when there is a grammar mistake in the query
    3. Above average: This is when the query is mostly correct but gets a logical step wrong in the query
    4. Bad: Makes more than one mistake in the query 
    5. Very bad: does not produce a query or varies significantly from the correct query



    Queries to grade:
    {queries_to_rank}

    {format_instructions}

    Return ONLY the JSON with grades list."""


    eval_template_sql = """Compare these SQL queries to the correct query and grade each one as: 'Very bad', 'Bad', 'Above average', 'Good', or 'Excellent'.
This is the following grading system, use the correct query as reference:

- Correct Query: {true_query}

1. Excellent: Only given when the SQL query is perfect and matches {true_query}, including:
   - Correct table names and relationship types
   - Proper SQL syntax and structure
   - Correct column naming and data access
   - Exact match of all conditions and return values

2. Good: When there are minor issues such as:
   - Small syntax variations (e.g., different column names but correct structure)
   - Equivalent but differently structured queries
   - Minor differences in how conditions or joins are constructed
   - Different but equivalent WHERE conditions

3. Above average: When the query has the right idea but contains:
   - Incorrect JOIN relationships or directions
   - Missing or extra tables
   - Incomplete filtering or grouping criteria
   - Partially incorrect conditions in WHERE or HAVING clauses
   - Different but related return columns

4. Bad: When the query has multiple issues:
   - Wrong query structure (e.g., incorrect JOIN usage)
   - Missing crucial JOINs or conditions
   - Incorrect or missing WHERE clauses
   - Wrong aggregations or GROUP BY usage
   - Incorrect use of SQL-specific features (subqueries, CTEs, etc.)

5. Very bad: When the query:
   - Is not a valid SQL query
   - Uses completely wrong structures or keywords
   - Returns entirely different data
   - Contains fundamental misunderstandings of SQL syntax
   - Is missing or has severe syntax errors
   
Queries to grade:
{queries_to_rank}
{format_instructions}
Return ONLY the JSON with grades list."""

    rewards = [0.0] * len(completions)
    
    for prompt, group in query_groups.items():
        try:
            num_queries = len(group)
            queries_text = "\n\n".join([f"Query {i+1}:\n{q['pred_query']}" 
                                      for i, q in enumerate(group)])
            
            eval_prompt = PromptTemplate(
                template=eval_template_sql if group[0]['single_query_type'].lower() == 'sql' else eval_template_cypher,
                input_variables=["queries_to_rank", "true_query", "num_queries"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            eval_chain = eval_prompt | llm | parser
            
            eval_result = eval_chain.invoke({
                "queries_to_rank": queries_text,
                "true_query": group[0]['true_query'],
                "num_queries": num_queries
            })
            
            for i, grade in enumerate(eval_result.grades):
                index = group[i]['index']
                rewards[index] = grade_rewards[grade]
                
                print(f"\nQuery {i+1}: {group[i]['pred_query']}")
                print(f"Grade: {grade}")
                print(f"classes Reward: {rewards[index]}")
                
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            for query_info in group:
                rewards[query_info['index']] = 0.0
    
    return rewards







