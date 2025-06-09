import json
import numpy as np
import re
import time
from typing import List, Dict, Any, Tuple, Optional
import openai


class ToolMatcher:
    def __init__(self, embedding_model: str = "text-embedding-3-large", top_servers: int = 5, top_tools: int = 3):
        """
        Setup the tool matcher
        
        Args:
            embedding_model: The name of the embedding model
            top_servers: The number of servers in the first stage
            top_tools: The number of tools to return
        """
        self.embedding_model = embedding_model
        self.top_servers = top_servers
        self.top_tools = top_tools
        self.servers_data = None
        self.tool_assistant_pattern = re.compile(
            r'<tool_assistant>\s*server:\s*(.*?)\s*tool:\s*(.*?)\s*</tool_assistant>',
            re.DOTALL
        )
        self.openai_client = None
        
    def load_data(self, data_path: str) -> None:
        """
        Load the tool data
        
        Args:
            data_path: The path to the JSON file containing the tool embeddings
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.servers_data = json.load(f)
            print(f"Loaded {len(self.servers_data)} servers from {data_path}")
        except Exception as e:
            raise ValueError(f"Error loading tool data: {e}")
    
    def setup_openai_client(self, base_url: str, api_version: str, api_key: str) -> None:
        """
        Setup the OpenAI client
        
        Args:
            base_url: The base URL of the OpenAI API
            api_version: The version of the OpenAI API
            api_key: The API key
        """
        self.openai_client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            api_version=api_version,
            api_key=api_key,
        )
    
    def extract_tool_assistant(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the tool assistant tag content from the text
        
        Args:
            text: The input text
        
        Returns:
            The extracted server and tool descriptions, or None, None if not found
        """
        match = self.tool_assistant_pattern.search(text)
        if match:
            server_desc = match.group(1).strip()
            tool_desc = match.group(2).strip()
            return server_desc, tool_desc
        return None, None
    
    def get_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """
        Get the embedding of the text
        
        Args:
            text: The input text
            max_retries: The maximum number of retries
        
        Returns:
            The embedding of the text, or None if failed
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Call setup_openai_client first.")
        
        for attempt in range(max_retries):
            try:
                time.sleep(0.05)  # Avoid frequent requests
                response = self.openai_client.embeddings.create(
                    input=[text],
                    model=self.embedding_model
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff strategy
                    print(f"Error getting embedding, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to get embedding after {max_retries} attempts: {e}")
                    return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate the cosine similarity between two vectors
        
        Args:
            vec1: The first vector
            vec2: The second vector
        
        Returns:
            The cosine similarity, ranging from [-1, 1], the closer to 1 the more similar
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Avoid division by zero errors caused by zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def match_servers(self, server_desc: str) -> List[Dict[str, Any]]:
        """
        Match the most relevant servers based on the server description
        
        Args:
            server_desc: The server description
        
        Returns:
            The top_servers servers, sorted by similarity in descending order
        """
        if not self.servers_data:
            raise ValueError("No server data loaded. Call load_data first.")
        
        # Get the query embedding
        query_embedding = self.get_embedding(server_desc)
        if not query_embedding:
            raise ValueError("Failed to get embedding for server description")
        
        # Calculate the description similarity for each server
        server_scores = []
        for server in self.servers_data:
            # First check if the server has a description embedding
            if "description_embedding" not in server:
                continue
            
            # Calculate the description similarity
            desc_similarity = self.cosine_similarity(
                query_embedding, 
                server["description_embedding"]
            )
            
            # If there is a summary embedding, also calculate the summary similarity
            summary_similarity = 0
            if "summary_embedding" in server:
                summary_similarity = self.cosine_similarity(
                    query_embedding, 
                    server["summary_embedding"]
                )
            
            # Take the maximum of the description and summary similarities as the final score
            final_score = max(desc_similarity, summary_similarity)
            
            server_scores.append({
                "server": server,
                "score": final_score
            })
        
        # Sort by similarity in descending order
        server_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Return the top_servers servers with the highest scores
        return server_scores[:self.top_servers]
    
    def match_tools(self, server_list: List[Dict[str, Any]], tool_desc: str) -> List[Dict[str, Any]]:
        """
        Find the most matching tools in the filtered servers
        
        Args:
            server_list: The list of servers
            tool_desc: The tool description
        
        Returns:
            The top_tools tools, sorted by similarity in descending order
        """
        # Get the query embedding
        query_embedding = self.get_embedding(tool_desc)
        if not query_embedding:
            raise ValueError("Failed to get embedding for tool description")
        
        # Collect all tools and calculate the similarity
        tool_scores = []
        
        for server_info in server_list:
            server = server_info["server"]
            server_score = server_info["score"]
            
            # Check if the server has a tool list
            if "tools" not in server or not server["tools"]:
                continue
            
            for tool in server["tools"]:
                # Check if the tool has a description embedding
                if "description_embedding" not in tool:
                    continue
                
                # Calculate the tool description similarity
                tool_similarity = self.cosine_similarity(
                    query_embedding, 
                    tool["description_embedding"]
                )
                
                # Combine the server score and tool score as the final score
                final_score = (server_score * tool_similarity) * max(server_score, tool_similarity)
                
                tool_scores.append({
                    "server_name": server["name"],
                    "tool_name": tool["name"],
                    "tool_description": tool.get("description", ""),
                    "parameters": tool.get("parameter", {}),
                    "server_score": server_score,
                    "tool_score": tool_similarity,
                    "final_score": final_score
                })
        
        # Sort by final score in descending order
        tool_scores.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Return the top_tools tools with the highest scores
        return tool_scores[:self.top_tools]
    
    def match(self, input_text: str) -> Dict[str, Any]:
        """
        Extract the tool requirement from the input text and match the most relevant tools
        
        Args:
            input_text: The input text, which may contain <tool_assistant> tags
        
        Returns:
            The matching result, containing the matched servers and tools
        """
        # Extract the server and tool descriptions from the input
        server_desc, tool_desc = self.extract_tool_assistant(input_text)
        
        if not server_desc or not tool_desc:
            return {
                "success": False,
                "error": "No tool_assistant tag found or invalid format",
                "matched_servers": [],
                "matched_tools": []
            }
        
        try:
            # First stage: match servers
            matched_servers = self.match_servers(server_desc)
            
            # Second stage: match tools in the filtered servers
            matched_tools = self.match_tools(matched_servers, tool_desc)
            
            return {
                "success": True,
                "server_description": server_desc,
                "tool_description": tool_desc,
                "matched_servers": matched_servers,
                "matched_tools": matched_tools
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "server_description": server_desc,
                "tool_description": tool_desc,
                "matched_servers": [],
                "matched_tools": []
            }
