import sys
import json
import time
from typing import List, Dict

class GameState:
    """博弈树节点状态"""
    def __init__(self, node_id: int, value: int = None, is_max: bool = True):
        self.node_id = node_id
        self.value = value  # 叶子节点的值
        self.is_max = is_max  # 是否为MAX节点
        self.children = []  # 子节点列表
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.pruned = False  # 是否被剪枝
        self.visit_order = 0  # 访问顺序
        self.best_child = None  # 最佳子节点

def parse_game_tree(data: str) -> GameState:
    """
    解析博弈树输入（JSON格式）
    格式: {"tree_structure": {"is_maximizing_player": true, "children": [...]}}
    或: {"is_maximizing_player": true, "children": [...]}
    """
    try:
        tree_data = json.loads(data)
        
        # 如果包含tree_structure字段，则提取
        if 'tree_structure' in tree_data:
            tree_data = tree_data['tree_structure']
        
        return build_tree_from_dict(tree_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {e}")

def build_tree_from_dict(data: dict, is_max: bool = True, node_id: int = 0) -> GameState:
    """从字典构建博弈树"""
    if 'value' in data and data['value'] is not None:
        # 叶子节点
        return GameState(node_id, value=data['value'], is_max=is_max)
    
    # 非叶子节点
    is_max = data.get('is_maximizing_player', is_max)
    node = GameState(node_id, is_max=is_max)
    
    children_data = data.get('children', [])
    for i, child_data in enumerate(children_data):
        child_node = build_tree_from_dict(child_data, not is_max, node_id * 10 + i + 1)
        node.children.append(child_node)
    
    return node

def alpha_beta(node: GameState, alpha: float, beta: float, depth: int, 
               visit_counter: List[int], stats: Dict) -> float:
    """
    Alpha-Beta剪枝算法
    
    参数:
        node: 当前节点
        alpha: alpha值
        beta: beta值
        depth: 当前深度
        visit_counter: 访问计数器
        stats: 统计信息
    
    返回:
        节点的值
    """
    visit_counter[0] += 1
    node.visit_order = visit_counter[0]
    node.alpha = alpha
    node.beta = beta
    
    stats['nodes_visited'] += 1
    stats['max_depth'] = max(stats['max_depth'], depth)
    
    # 叶子节点
    if not node.children:
        stats['leaf_nodes_visited'] += 1
        return node.value
    
    if node.is_max:
        value = float('-inf')
        for i, child in enumerate(node.children):
            child_value = alpha_beta(child, alpha, beta, depth + 1, visit_counter, stats)
            value = max(value, child_value)
            alpha = max(alpha, value)
            
            # 记录最佳子节点
            if value == child_value:
                node.best_child = child
            
            # Alpha剪枝
            if alpha >= beta:
                stats['pruned_nodes'] += 1
                node.pruned = True
                # 标记剩余子节点为剪枝
                for remaining_child in node.children[i + 1:]:
                    remaining_child.pruned = True
                break
        return value
    else:
        value = float('inf')
        for i, child in enumerate(node.children):
            child_value = alpha_beta(child, alpha, beta, depth + 1, visit_counter, stats)
            value = min(value, child_value)
            beta = min(beta, value)
            
            # 记录最佳子节点
            if value == child_value:
                node.best_child = child
            
            # Beta剪枝
            if alpha >= beta:
                stats['pruned_nodes'] += 1
                node.pruned = True
                # 标记剩余子节点为剪枝
                for remaining_child in node.children[i + 1:]:
                    remaining_child.pruned = True
                break
        return value

def minimax_without_pruning(node: GameState, depth: int, 
                            visit_counter: List[int], stats: Dict) -> float:
    """不带剪枝的Minimax算法（用于对比）"""
    visit_counter[0] += 1
    stats['nodes_visited'] += 1
    stats['max_depth'] = max(stats['max_depth'], depth)
    
    if not node.children:
        stats['leaf_nodes_visited'] += 1
        return node.value
    
    if node.is_max:
        value = float('-inf')
        for child in node.children:
            child_value = minimax_without_pruning(child, depth + 1, visit_counter, stats)
            value = max(value, child_value)
        return value
    else:
        value = float('inf')
        for child in node.children:
            child_value = minimax_without_pruning(child, depth + 1, visit_counter, stats)
            value = min(value, child_value)
        return value

def print_tree(node: GameState, depth: int = 0, prefix: str = ""):
    """打印博弈树结构"""
    indent = "  " * depth
    node_type = "MAX" if node.is_max else "MIN"
    
    if node.children:
        print(f"{indent}{prefix}节点{node.node_id} ({node_type})")
        for i, child in enumerate(node.children):
            if child.pruned:
                print(f"{indent}  ├─ [剪枝] 节点{child.node_id}")
            else:
                print_tree(child, depth + 1, "├─ ")
    else:
        print(f"{indent}{prefix}叶子{node.node_id}: {node.value}")

def print_alpha_beta_trace(node: GameState, depth: int = 0, prefix: str = ""):
    """打印Alpha-Beta剪枝过程"""
    indent = "  " * depth
    node_type = "MAX" if node.is_max else "MIN"
    
    if node.children:
        alpha_str = f"α={node.alpha:.0f}" if node.alpha != float('-inf') else "α=-∞"
        beta_str = f"β={node.beta:.0f}" if node.beta != float('inf') else "β=∞"
        print(f"{indent}{prefix}节点{node.node_id} ({node_type}) [{alpha_str}, {beta_str}] 访问顺序:{node.visit_order}")
        for i, child in enumerate(node.children):
            if child.pruned:
                print(f"{indent}  ├─ [剪枝] 节点{child.node_id}")
            else:
                print_alpha_beta_trace(child, depth + 1, "├─ ")
    else:
        print(f"{indent}{prefix}叶子{node.node_id}: {node.value} (访问顺序:{node.visit_order})")

def print_stats(stats: Dict, title: str = "Alpha-Beta剪枝统计"):
    """打印统计信息"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f" 总访问节点数: {stats['nodes_visited']}")
    print(f" 叶子节点访问数: {stats['leaf_nodes_visited']}")
    print(f" 剪枝节点数: {stats['pruned_nodes']}")
    print(f" 最大深度: {stats['max_depth']}")
    print(f" 搜索时间: {stats['search_time']:.3f} ms")
    
    if stats['total_nodes'] > 0:
        prune_rate = stats['pruned_nodes'] / stats['total_nodes'] * 100
        print(f" 剪枝率: {prune_rate:.1f}%")
    
    print(f"{'='*50}")

def compare_with_minimax(root: GameState):
    """对比Alpha-Beta剪枝与普通Minimax"""
    print("\n" + "="*60)
    print(" Alpha-Beta剪枝 vs 普通Minimax 对比")
    print("="*60)
    
    # 运行Alpha-Beta
    ab_stats = {
        'nodes_visited': 0,
        'leaf_nodes_visited': 0,
        'pruned_nodes': 0,
        'max_depth': 0,
        'search_time': 0,
        'total_nodes': 0
    }
    
    # 计算总节点数
    def count_nodes(node):
        count = 1
        for child in node.children:
            count += count_nodes(child)
        return count
    ab_stats['total_nodes'] = count_nodes(root)
    
    visit_counter = [0]
    start_time = time.time()
    ab_result = alpha_beta(root, float('-inf'), float('inf'), 0, visit_counter, ab_stats)
    ab_stats['search_time'] = (time.time() - start_time) * 1000
    
    # 运行普通Minimax
    mm_stats = {
        'nodes_visited': 0,
        'leaf_nodes_visited': 0,
        'pruned_nodes': 0,
        'max_depth': 0,
        'search_time': 0,
        'total_nodes': 0
    }
    mm_stats['total_nodes'] = count_nodes(root)
    
    visit_counter = [0]
    start_time = time.time()
    mm_result = minimax_without_pruning(root, 0, visit_counter, mm_stats)
    mm_stats['search_time'] = (time.time() - start_time) * 1000
    
    # 打印结果
    print(f"\n{'指标':<20} {'Alpha-Beta':<15} {'普通Minimax':<15}")
    print("-"*50)
    print(f"{'结果值':<20} {ab_result:<15} {mm_result:<15}")
    print(f"{'访问节点数':<20} {ab_stats['nodes_visited']:<15} {mm_stats['nodes_visited']:<15}")
    print(f"{'叶子节点访问':<20} {ab_stats['leaf_nodes_visited']:<15} {mm_stats['leaf_nodes_visited']:<15}")
    print(f"{'搜索时间(ms)':<20} {ab_stats['search_time']:<15.3f} {mm_stats['search_time']:<15.3f}")
    
    # 效率提升
    if mm_stats['nodes_visited'] > 0:
        reduction = (mm_stats['nodes_visited'] - ab_stats['nodes_visited']) / mm_stats['nodes_visited'] * 100
        print(f"\n 节点访问减少率: {reduction:.1f}%")
    
    print(f"\n 结果一致性: {'✓ 一致' if ab_result == mm_result else '✗ 不一致'}")
    print(f"{'='*60}")

def main():
    """主程序"""
    print("="*60)
    print(" 题目三：Alpha-Beta剪枝算法")
    print("="*60)
    
    print("\n请输入博弈树数据（JSON格式）：")
    print("格式: {'tree_structure': {'is_maximizing_player': true, 'children': [...]}}")
    print("或: {'is_maximizing_player': true, 'children': [...]}")
    print("提示：输入完成后先按回车，再按Ctrl+D（Linux/Mac）或Ctrl+Z（Windows），再按回车结束")
    
    # 读取输入
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:
            break
    
    data = '\n'.join(lines).strip()
    
    if not data:
        print("输入为空，使用默认测试用例")
        data = '''{
            "tree_structure": {
                "is_maximizing_player": true,
                "children": [
                    {
                        "is_maximizing_player": false,
                        "children": [
                            {
                                "is_maximizing_player": true,
                                "children": [
                                    {"value": 3},
                                    {"value": 5}
                                ]
                            },
                            {
                                "is_maximizing_player": true,
                                "children": [
                                    {"value": 6},
                                    {"value": 9}
                                ]
                            }
                        ]
                    },
                    {
                        "is_maximizing_player": false,
                        "children": [
                            {
                                "is_maximizing_player": true,
                                "children": [
                                    {"value": 6},
                                    {"value": 9}
                                ]
                            },
                            {
                                "is_maximizing_player": true,
                                "children": [
                                    {"value": 1},
                                    {"value": 2}
                                ]
                            }
                        ]
                    }
                ]
            }
        }'''
    
    # 解析博弈树
    try:
        root = parse_game_tree(data)
    except Exception as e:
        print(f"解析错误: {e}")
        return
    
    # 打印博弈树结构
    print("\n" + "="*60)
    print(" 博弈树结构")
    print("="*60)
    print_tree(root)
    
    # 运行Alpha-Beta剪枝
    print("\n" + "="*60)
    print(" Alpha-Beta剪枝过程")
    print("="*60)
    
    stats = {
        'nodes_visited': 0,
        'leaf_nodes_visited': 0,
        'pruned_nodes': 0,
        'max_depth': 0,
        'search_time': 0,
        'total_nodes': 0
    }
    
    # 计算总节点数
    def count_nodes(node):
        count = 1
        for child in node.children:
            count += count_nodes(child)
        return count
    stats['total_nodes'] = count_nodes(root)
    
    visit_counter = [0]
    start_time = time.time()
    result = alpha_beta(root, float('-inf'), float('inf'), 0, visit_counter, stats)
    stats['search_time'] = (time.time() - start_time) * 1000
    
    print(f"\n最终结果值: {result}")
    
    # 打印剪枝过程
    print("\n" + "="*60)
    print(" Alpha-Beta剪枝详细过程")
    print("="*60)
    print_alpha_beta_trace(root)
    
    # 打印统计信息
    print_stats(stats)
    
    # 对比普通Minimax
    print("\n是否进行Minimax对比？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        compare_with_minimax(root)

if __name__ == "__main__":
    main()