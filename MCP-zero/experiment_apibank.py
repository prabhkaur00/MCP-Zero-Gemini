#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import sys
import argparse
import datetime
import csv
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# 添加上级目录到系统路径，以便导入call_openai模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from call_openai.function_call_gpt import chat_any
from call_openai.call_embed import embeddings

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_system_prompt(file_path: str) -> str:
    """
    Load system prompt template from file
    
    Args:
        file_path: Path to system prompt template file
    
    Returns:
        System prompt template content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_tool_embeddings(file_path: str) -> List[Dict[str, Any]]:
    """
    Load tool embeddings data from file
    
    Args:
        file_path: Path to tool embeddings file
    
    Returns:
        Tool embeddings data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_tool_description(response: str) -> str:
    """
    Extract tool description from model response
    
    Args:
        response: Model response containing tool description
    
    Returns:
        Extracted tool description
    """
    # Use regex to extract content between <tool_assistant> tags
    pattern = r'<tool_assistant>(.*?)</tool_assistant>'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        
        # Extract description after "tool:" if present
        tool_pattern = r'tool:\s*(.*?)$'
        tool_match = re.search(tool_pattern, content, re.DOTALL)
        
        if tool_match:
            return tool_match.group(1).strip()
        else:
            return content.strip()
    
    return response.strip()

def get_embedding_for_description(description: str) -> List[float]:
    """
    Get embedding vector for a tool description
    
    Args:
        description: Tool description
    
    Returns:
        Embedding vector
    """
    try:
        return embeddings(description)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return empty vector in case of error
        return [0.0] * 3072  # Assuming embedding size is 3072

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Cosine similarity score
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)

def find_most_similar_tool(description_embedding: List[float], 
                           tool_embeddings: List[Dict[str, Any]],
                           tools_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    """
    Find the most similar tool based on embedding similarity
    
    Args:
        description_embedding: Embedding of the description
        tool_embeddings: List of tool embeddings with global tool information
        tools_data: List of available tools for this specific test case
        
    Returns:
        Most similar tool and similarity score
    """
    max_similarity = -1.0
    most_similar_tool = None
    
    # Create a mapping from tool_name to its embedding
    tool_embedding_map = {tool.get("tool_name", ""): tool.get("embedding", []) 
                          for tool in tool_embeddings if tool.get("tool_name")}
    
    # Only match against tools available in this test case
    for tool in tools_data:
        tool_name = tool.get("name", "")
        
        # # Skip if tool name not found in embeddings
        if tool_name not in tool_embedding_map:
            continue
        # match all tools
            
        tool_embedding = tool_embedding_map[tool_name]
        
        if not tool_embedding:
            continue
        
        similarity = cosine_similarity(description_embedding, tool_embedding)
        
        if similarity > max_similarity:
            max_similarity = similarity
            # Create a result with both embedding and tool info
            most_similar_tool = {
                "tool_name": tool_name,
                "description": tool.get("description", ""),
                "embedding": tool_embedding
            }
    
    return most_similar_tool, max_similarity

def test_tool_retrieval(system_prompt: str, 
                    #    tools_data: List[Dict[str, Any]],
                       user_query: str,
                       model_name: str = "gcp-claude35-sonnet",
                       verbose: bool = False) -> str:
    """
    Test the tool retrieval capability of the model
    
    Args:
        system_prompt: System prompt
        user_query: User query
        model_name: Model name
        verbose: Whether to print detailed information
        
    Returns:
        Model response
    """
    # Print the complete system prompt if verbose
    if verbose:
        print("\n===== SYSTEM PROMPT =====")
        print(system_prompt)
        print("=========================\n")
    
    # Call the model
    try:
        response = chat_any(system_prompt, user_query, model_name)
        return response.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return f"ERROR: {str(e)}"

def extract_target_function_from_example(example: Dict[str, Any]) -> str:
    """
    Extract the target function name from an example
    
    Args:
        example: Example data
        
    Returns:
        Target function name
    """
    return example.get("output_tool_name", "")

def get_test_cases_from_data(api_data: List[Dict[str, Any]], num_examples: int = None) -> List[Dict[str, Any]]:
    """
    Extract test cases from API data
    
    Args:
        api_data: API data
        num_examples: Number of examples to use (None for all)
        
    Returns:
        List of test cases
    """
    test_cases = []
    
    # Limit to specified number of examples if provided
    data_to_use = api_data[:num_examples] if num_examples is not None else api_data
    
    for example in data_to_use:
        # Get input dialog and expected output
        input_dialog = example.get("input", "")
        target_function = extract_target_function_from_example(example)
        tools_data = example.get("tools", [])
        
        # # Extract the last user query from input dialog
        # lines = input_dialog.strip().split("\n")
        # user_query = ""
        # for line in reversed(lines):
        #     if line.startswith("User:"):
        #         user_query = line[6:].strip()
        #         break
        user_query = input_dialog
        
        if user_query and target_function and tools_data:
            test_cases.append({
                "user_query": user_query,
                "target_function": target_function,
                "tools_data": tools_data,
                "example_id": example.get("id", 0)
            })
    
    return test_cases

def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test tool retrieval capabilities')
    parser.add_argument('--model', type=str, default="gcp-claude35-sonnet", 
                        help='Model to use (default: gcp-claude35-sonnet)')
    parser.add_argument('--query', type=str, 
                        help='Custom user query to test (single test mode)')
    parser.add_argument('--data', type=str, default="single_turn.json",     # 单轮对话
    # parser.add_argument('--data', type=str, default="multi_turn.json",    # 多轮对话
                        help='Path to API data file')
    parser.add_argument('--prompt', type=str, default="../prompts/system_ours_apibank.prompt",
    # parser.add_argument('--prompt', type=str, default="../prompts/system_ours_apibank_icl.prompt",
                        help='Path to system prompt template file')
    parser.add_argument('--embeddings', type=str, default="apibank_tool_embeddings.json",
                        help='Path to tool embeddings file')
    parser.add_argument('--index', type=int,
                        help='Index of the example to use from the API data (single test mode)')
    parser.add_argument('--batch', action='store_true',
                        help='Run batch testing on all examples')
    parser.add_argument('--limit', type=int,
                        help='Limit number of examples to test in batch mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    args = parser.parse_args()
    
    # Set file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_data_path = os.path.join(script_dir, args.data)
    system_prompt_path = os.path.join(script_dir, args.prompt)
    tool_embeddings_path = os.path.join(script_dir, args.embeddings)
    
    # Load data
    api_data = load_json_data(api_data_path)
    tool_embeddings = load_tool_embeddings(tool_embeddings_path)
    
    # Load system prompt
    system_prompt = load_system_prompt(system_prompt_path)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Batch testing mode
    if args.batch:
        print(f"Starting batch testing with model: {args.model}")
        print(f"Using up to {args.limit if args.limit else 'all'} examples from {api_data_path}")
        
        # Get test cases from data
        test_cases = get_test_cases_from_data(api_data, args.limit)
        print(f"Found {len(test_cases)} valid test cases")
        
        # Initialize results tracking
        results = []
        exact_match_count = 0
        similarity_sum = 0.0
        total_count = len(test_cases)
        
        # Run tests
        for i, test_case in enumerate(test_cases):
            example_id = test_case["example_id"]
            user_query = test_case["user_query"]
            target_function = test_case["target_function"]
            tools_data = test_case["tools_data"]
            
            print(f"\nTesting example {i+1}/{total_count} (ID: {example_id})")
            # print(f"Query: {user_query}")
            print(f"Target function: {target_function}")
            
            time.sleep(1.5)  # Rate limiting
            
            # Run test
            model_response = test_tool_retrieval(
                system_prompt, 
                # tools_data,
                user_query, 
                args.model,
                args.verbose
            )
            
            # Extract tool description
            tool_description = extract_tool_description(model_response)
            print(f"Extracted description: {tool_description}")
            
            Get embedding for the description
            
            # # [baseline experiment]: use the user query as the description
            # model_response = user_query
            
            description_embedding = get_embedding_for_description(model_response)
            
            # Find most similar tool
            most_similar_tool, similarity = find_most_similar_tool(
                description_embedding, 
                tool_embeddings,
                tools_data
            )
            
            predicted_function = most_similar_tool.get("tool_name", "") if most_similar_tool else ""
            
            # Check if prediction is correct
            is_exact_match = predicted_function == target_function
            if is_exact_match:
                exact_match_count += 1
            
            similarity_sum += similarity
            
            # Record result
            print(f"Predicted function: {predicted_function}")
            # print(f"Similarity score: {similarity:.4f}")
            print(f"Exact match: {is_exact_match}")
            
            results.append({
                "example_id": example_id,
                "user_query": user_query,
                "target_function": target_function,
                "model_response": user_query,
                "extracted_description": user_query,
                "predicted_function": predicted_function,
                "similarity_score": similarity,
                "is_exact_match": is_exact_match
            })
        
        # Calculate metrics
        exact_match_rate = exact_match_count / total_count if total_count > 0 else 0
        avg_similarity = similarity_sum / total_count if total_count > 0 else 0
        
        print(f"\nResults Summary:")
        print(f"Exact matches: {exact_match_count}/{total_count} = {exact_match_rate:.2%}")
        print(f"Average similarity score: {avg_similarity:.4f}")
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(results_dir, f"retrieval_results_{args.model}_{timestamp}.json")
        
        # Prepare results dictionary
        results_dict = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": args.model,
            "total_tests": total_count,
            "exact_matches": exact_match_count,
            "exact_match_rate": exact_match_rate,
            "average_similarity": avg_similarity,
            "results": results
        }
        
        # Save to JSON file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
        print(f"Results saved to {results_file}")
        
if __name__ == "__main__":
    main() 