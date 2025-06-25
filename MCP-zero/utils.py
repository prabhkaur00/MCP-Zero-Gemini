#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple


def generate_grid_search_params(
    num_position_ratios: int = 20,
    num_sample_sizes: int = 50,
    total_tools: int = 2797
) -> Tuple[List[float], List[int]]:
    """
    生成网格搜索参数
    
    Args:
        num_position_ratios: 位置比例的数量
        num_sample_sizes: 样本大小的数量
        total_tools: 总工具数量
    
    Returns:
        (位置比例列表, 样本大小列表)
    """
    # 生成位置比例列表 [0.0, 0.05, 0.1, ..., 1.0]
    position_ratios = np.linspace(0, 1, num_position_ratios+1).tolist()
    
    # 生成样本大小列表，对数均匀分布 [1, 2, 4, 8, ..., 2797]
    # 使用对数空间进行均匀分割，然后取指数得到实际的样本大小
    log_space = np.linspace(np.log(1), np.log(total_tools), num_sample_sizes)
    sample_sizes = [max(1, int(np.round(np.exp(x)))) for x in log_space]
    
    # 生成位置索引列表: 样本数量 * 位置比例，取整
    position_indexes = []
    for sample_size in sample_sizes:
        for position_ratio in position_ratios:
            # 去重
            target = [int((sample_size - 1) * position_ratio), sample_size]
            if target not in position_indexes:
                position_indexes.append(target)
    
    return position_indexes


if __name__ == "__main__":
    grid_points_list = generate_grid_search_params(num_position_ratios=2, num_sample_sizes=3)
    print(grid_points_list)