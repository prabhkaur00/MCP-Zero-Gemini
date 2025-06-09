import json
from typing import List, Dict, Any

from .matcher import ToolMatcher


def load_test_cases(jsonl_path: str) -> List[Dict[str, Any]]:
    test_cases = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_case = json.loads(line)
                    test_cases.append(test_case)
        print(f"Loading {len(test_cases)} test cases")
        return test_cases
    except Exception as e:
        print(f"Error loading test cases: {e}")
        return []


def print_match_result(result: Dict[str, Any]) -> None:
    if not result["success"]:
        print(f"Matching failed: {result.get('error', 'Unknown error')}")
        return
    
    print(f"Server description: {result['server_description']}")
    print(f"Tool description: {result['tool_description']}")
    
    print("\nMatched servers (top 3):")
    for i, server_info in enumerate(result["matched_servers"]):
        print(f"{i+1}. {server_info['server']['name']} (Score: {server_info['score']:.4f})")
        print(f"   Description: {server_info['server'].get('description', 'No description')}")
        if "summary" in server_info['server']:
            print(f"   Summary: {server_info['server']['summary']}")
    
    print("\nMatched tools (top 3):")
    for i, tool_info in enumerate(result["matched_tools"]):
        print(f"{i+1}. {tool_info['server_name']} -> {tool_info['tool_name']} (Final score: {tool_info['final_score']:.4f})")
        print(f"   Description: {tool_info['tool_description']}")



def main():
    matcher = ToolMatcher(top_servers=3, top_tools=3)

    data_path = "mcp-tools/mcp_tools_with_embedding.json"
    test_cases_path = "mcp-zero/test_cases.jsonl"
    matcher.load_data(data_path)
    
    # Setup OpenAI client
    base_url = ""
    api_version = ""
    api_key = ""
    matcher.setup_openai_client(base_url, api_version, api_key)
    
    # Load test cases
    test_cases = load_test_cases(test_cases_path)
    
    # Run all test cases
    for i, test_case in enumerate(test_cases):
        test_input = test_case.get("input", "")
        result = matcher.match(test_input)
        print_match_result(result)


if __name__ == "__main__":
    main()
