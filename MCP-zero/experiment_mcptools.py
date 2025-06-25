#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import datetime
import re
from typing import List, Dict, Any, Tuple, Optional, Union


from sampler import ToolSampler
from matcher import ToolMatcher
from reformatter import format_tools_as_functions
from call_openai.function_call_gpt import (
    chat_gpt4_1,
    chat_claude3_5
)
from utils import generate_grid_search_params



def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_tool_assistant(text: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = re.compile(
        r'<tool_assistant>\s*server:\s*(.*?)\s*tool:\s*(.*?)\s*</tool_assistant>',
        re.DOTALL
    )
    match = pattern.search(text)
    if match:
        server_desc = match.group(1).strip()
        tool_desc = match.group(2).strip()
        return server_desc, tool_desc
    return None, None


def test_llm_retrieval(
    sampled_data: List[Dict[str, Any]],
    target_server: Dict[str, Any],
    target_tool: Dict[str, Any],
    sample_size: int = 20,
    position_index: int = 0,
    use_random_selection: bool = False,
    output_dir: str = None,
    model_name: str = "gpt-4.1"
) -> Tuple[Dict[str, Any], Optional[int]]:
    """
    
    Args:
        sampled_data: 采样的数据
        target_server: 目标服务器信息
        target_tool: 目标工具信息
        sample_size: 采样工具数量
        position_index: 目标工具的位置索引
        use_random_selection: 是否使用随机选择
        output_dir: 输出目录路径
        model_name: 模型名称
    
    Returns:
        (测试结果, None)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
    # 设置结果文件名的后缀
    if use_random_selection:
        selection_method = "random"
    else:
        selection_method = f"position_index_{position_index}"
    
    # 读取系统提示词
    system_prompt_path = os.path.join(output_dir, "ours_system_prompt.txt")
    system_prompt = read_text_file(system_prompt_path)
    
    # 读取用户提示词模板
    user_prompt_template_path = os.path.join(output_dir, "ours_user_prompt_template.txt")
    user_prompt_template = read_text_file(user_prompt_template_path)
    
    # 获取目标服务器和工具描述
    server_description = target_server.get("description", "")
    tool_description = target_tool.get("description", "")
    
    # 构建用户提示词
    user_prompt = user_prompt_template.replace("{server_description}", server_description)
    user_prompt = user_prompt.replace("{tool_description}", tool_description)
    
    # 开始计时
    start_time = time.time()
    
    # 调用大模型
    try:
        if model_name.lower() == "gpt-4.1":
            response = chat_gpt4_1(system_prompt, user_prompt)
        else:
            response = chat_claude3_5(system_prompt, user_prompt)
        success = True
    except Exception as e:
        print(f"Error: {str(e)}")
        response = f"Error: {str(e)}"
        success = False
    
    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 从响应中提取服务器描述和工具描述
    extracted_server_desc, extracted_tool_desc = extract_tool_assistant(response)
    
    # 初始化匹配结果
    is_correct = False
    matched_server = None
    matched_tool = None
    
    TOP_SERVERS = 5
    TOP_TOOLS = 1
    
    # 如果成功提取了描述，进行向量匹配
    if extracted_server_desc and extracted_tool_desc:
        # 初始化工具匹配器
        matcher = ToolMatcher(top_servers=TOP_SERVERS, top_tools=TOP_TOOLS)
        
        # 设置OpenAI客户端 (使用与data/mcp-tools相同的客户端配置)
        base_url = ""
        api_version = ""
        api_key = ""
        matcher.setup_openai_client(base_url, api_version, api_key)
        
        # 准备匹配结果
        match_result = {
            "success": True,
            "server_description": extracted_server_desc,
            "task_description": extracted_tool_desc,
            "matched_servers": [],
            "matched_tools": []
        }
        
        try:
            query_server_embedding = matcher.get_embedding(extracted_server_desc)
            query_tool_embedding = matcher.get_embedding(extracted_tool_desc)
            
            if not query_server_embedding or not query_tool_embedding:
                raise ValueError("Failed to get embeddings for server or tool description")
            
            # 使用ToolMatcher进行分层匹配
            # 第一阶段：匹配服务器
            server_scores = []
            for server in sampled_data:
                # 检查服务器是否有描述嵌入
                if "description_embedding" not in server:
                    continue
                if "tools" not in server or not server["tools"]:
                    continue
                
                # 计算描述相似度
                desc_similarity = matcher.cosine_similarity(
                    query_server_embedding, 
                    server["description_embedding"]
                )
                
                # 如果有摘要嵌入，也计算摘要相似度
                summary_similarity = 0
                if "summary_embedding" in server:
                    summary_similarity = matcher.cosine_similarity(
                        query_server_embedding, 
                        server["summary_embedding"]
                    )
                
                # 取描述和摘要相似度的最大值作为最终得分
                final_score = max(desc_similarity, summary_similarity)
                
                server_scores.append({
                    "server": server,
                    "score": final_score
                })
            
            # 按相似度降序排序
            server_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # 取前5个服务器（与matcher.py中的top_servers保持一致）
            matched_servers = server_scores[:TOP_SERVERS]
            match_result["matched_servers"] = matched_servers
            
            # 第二阶段：在筛选出的服务器中匹配工具
            tool_scores = []
            
            for server_info in matched_servers:
                server = server_info["server"]
                server_score = server_info["score"]

                for tool in server["tools"]:
                    # 检查工具是否有描述嵌入
                    if "description_embedding" not in tool:
                        continue
                    
                    # 计算工具描述相似度
                    tool_similarity = matcher.cosine_similarity(
                        query_tool_embedding, 
                        tool["description_embedding"]
                    )
                    
                    # 最终得分结合服务器得分和工具得分
                    # 使用与matcher.py相同的得分计算方式
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
            
            # 按最终得分降序排序
            tool_scores.sort(key=lambda x: x["final_score"], reverse=True)
            
            # 取前1个工具（与matcher.py中的top_tools保持一致）
            matched_tools = tool_scores[:TOP_TOOLS]
            match_result["matched_tools"] = matched_tools
            
            # 如果有匹配结果，检查是否匹配到目标工具:只对比 top-1
            if matched_tools:
                matched_tool = matched_tools[0]
                matched_server = next((s for s in matched_servers if s["server"]["name"] == matched_tool["server_name"]), None)
                
                # 检查是否匹配到了目标工具
                is_correct = (
                    matched_tool["server_name"] == target_server.get("name", "") and
                    matched_tool["tool_name"] == target_tool.get("name", "")
                )
                
        except Exception as e:
            print(f"Error: {str(e)}")
            match_result["success"] = False
            match_result["error"] = str(e)
    
    # 构建最终结果
    result = {
        "success": success,
        "elapsed_time": elapsed_time,
        "response": response.strip(),
        "extracted_server_desc": extracted_server_desc,
        "extracted_tool_desc": extracted_tool_desc,
        "is_correct": is_correct,
        "target_server_name": target_server.get("name", ""),
        "target_tool_name": target_tool.get("name", ""),
        "target_server_description": server_description,
        "target_tool_description": tool_description,
        "matched_server": matched_server["server"]["name"] if matched_server else None,
        "matched_tool": matched_tool["tool_name"] if matched_tool else None,
        "sample_size": sample_size,
        "position_index": position_index,
        "selection_method": selection_method,
    }
    
    return result, None


def run_grid_search(
    data_path: str,
    output_dir: str = None,
    num_position_ratios: int = 20,
    num_sample_sizes: int = 50,
    request_interval: float = 5.0,
) -> None:
    """
    运行网格搜索测试
    
    Args:
        data_path: 数据文件路径
        output_dir: 输出目录路径
        num_position_ratios: 位置等分，实际上会多一个
        num_sample_sizes: 样本大小的数量
        request_interval: 请求间隔时间（秒）
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 生成网格搜索参数
    grid_points_list = generate_grid_search_params(num_position_ratios, num_sample_sizes)
    
    # 创建结果目录
    results_dir = os.path.join(output_dir, "grid_search_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化结果列表
    all_results = []
    
    # 结果文件路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"ours_grid_search_results_{timestamp}.json")
    
    # 初始化采样器
    sampler = ToolSampler(data_path)
    
    # 缓存已经采样的数据
    sampled_data_cache = {}
    
    for (position_index, sample_size) in grid_points_list:
        print(f"\n=== 处理样本: {position_index} / {sample_size} ===")
        
        # 采样工具（或从缓存获取）
        if sample_size not in sampled_data_cache:
            sampled_data = sampler.sample_tools(sample_size)
            sampled_data_cache[sample_size] = sampled_data
        else:
            sampled_data = sampled_data_cache[sample_size]
        
        # 选择目标工具
        target_server, target_tool = sampler.select_target_tool(sampled_data, position_index)
        
        # 测试大模型的工具检索能力
        result, _ = test_llm_retrieval(
            sampled_data=sampled_data,
            target_server=target_server,
            target_tool=target_tool,
            sample_size=sample_size,
            position_index=position_index,
            use_random_selection=False,
            output_dir=output_dir,
            model_name="claude3.5"  # 可以选择使用 "claude3.5"
        )
        
        # 添加到结果列表
        all_results.append(result)
        
        # 保存当前进度
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 打印结果
        print(f"  结果: {'✓' if result['is_correct'] else '✗'} "
              f"耗时: {result['elapsed_time']:.2f}秒")
        print(f"  目标服务器: {result['target_server_name']}, 目标工具: {result['target_tool_name']}")
        print(f"  提取的服务器描述: {result['extracted_server_desc']}")
        print(f"  提取的工具描述: {result['extracted_tool_desc']}")
        print(f"  匹配的服务器: {result['matched_server']}, 匹配的工具: {result['matched_tool']}")
        
        # 等待一段时间，避免请求过于频繁
        if position_index != len(grid_points_list) - 1 or sample_size != len(grid_points_list) - 1:
            print(f"  等待 {request_interval} 秒...")
            time.sleep(request_interval)
    
    print(f"\n=== 网格搜索完成 ===")
    print(f"总共测试了 {len(all_results)} 个配置")
    print(f"结果已保存到: {results_file}")
    
    # 计算正确率
    correct_count = sum(1 for result in all_results if result["is_correct"])
    accuracy = correct_count / len(all_results) if all_results else 0
    print(f"总体正确率: {accuracy:.2%}")


if __name__ == "__main__":
    # 默认数据路径
    data_path = "./mcp-tools/all_servers_filtered_with_embedding.json"
    
    # 运行网格搜索
    run_grid_search(
        data_path=data_path,
        num_position_ratios=20,  # 位置等分数量
        num_sample_sizes=50,     # 样本大小数量
        request_interval=3.0,    # 请求间隔2秒
    ) 