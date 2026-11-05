import heapq
import sys
import ast
import time
from typing import List, Tuple, Dict, Set

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """曼哈顿距离"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_maze_with_stats(start: Tuple[int, int], 
                           end: Tuple[int, int], 
                           grid: List[List[str]],
                           heuristic='manhattan') -> Tuple[List[Tuple[int, int]], Dict]:
    """
    A*算法求解迷宫最短路径
    
    参数:
        heuristic: 'manhattan', 'euclidean', 'zero' 
    
    返回:
        (路径, 统计信息)
    """
    rows, cols = len(grid), len(grid[0])
    
    # 统计信息
    stats = {
        'nodes_expanded': 0,
        'nodes_visited': 0,
        'nodes_in_open': 0,
        'nodes_in_closed': 0,
        'max_f_value': 0,
        'min_f_value': float('inf'),
        'avg_f_value': 0,
        'total_f_sum': 0,
        'path_length': 0,
        'search_time': 0,
        'memory_usage': 0,
        'branching_factor': 0,
        'solution_depth': 0,
        'effective_branching': 0,
        'optimal_cost': 0,
    }
    
    # 选择启发函数
    def get_h(node):
        if heuristic == 'manhattan':
            return manhattan(node, end)
        elif heuristic == 'euclidean':
            return ((node[0] - end[0])**2 + (node[1] - end[1])**2) ** 0.5
        elif heuristic == 'zero':
            return 0
        elif heuristic == 'diagonal':
            dx = abs(node[0] - end[0])
            dy = abs(node[1] - end[1])
            return max(dx, dy) + 0.414 * min(dx, dy)
        else:
            return manhattan(node, end)
    
    start_time = time.time()
    
    # 优先队列: (f值, g值, 坐标)
    open_set = [(get_h(start), 0, start)]
    g_score = {start: 0}
    came_from = {}
    closed_set = set()
    
    f_values = []
    open_set_size = 1
    
    while open_set:
        open_set_size = max(open_set_size, len(open_set))
        
        f, g, current = heapq.heappop(open_set)
        stats['nodes_expanded'] += 1
        
        f_values.append(f)
        stats['max_f_value'] = max(stats['max_f_value'], f)
        stats['min_f_value'] = min(stats['min_f_value'], f)
        stats['total_f_sum'] += f
        
        if current in closed_set:
            continue
            
        if current == end:
            # 回溯路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]  # 反转，确保从起点到终点
            
            end_time = time.time()
            stats['search_time'] = (end_time - start_time) * 1000
            stats['path_length'] = len(path) - 1
            stats['nodes_in_closed'] = len(closed_set)
            stats['nodes_in_open'] = open_set_size
            stats['memory_usage'] = len(g_score) + len(came_from) + len(closed_set)
            stats['avg_f_value'] = stats['total_f_sum'] / len(f_values) if f_values else 0
            stats['branching_factor'] = stats['nodes_visited'] / stats['nodes_expanded'] if stats['nodes_expanded'] > 0 else 0
            stats['solution_depth'] = len(path) - 1
            stats['effective_branching'] = stats['nodes_expanded'] ** (1 / max(1, stats['solution_depth']))
            stats['optimal_cost'] = g_score[end]
            
            return path, stats
        
        closed_set.add(current)
        
        # 四个方向移动
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = current[0] + dr, current[1] + dc
            
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
                
            if grid[nr][nc] == 'X':
                continue
                
            neighbor = (nr, nc)
            tentative_g = g_score[current] + 1
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                h = get_h(neighbor)
                f = tentative_g + h
                heapq.heappush(open_set, (f, tentative_g, neighbor))
                came_from[neighbor] = current
                stats['nodes_visited'] += 1
    
    end_time = time.time()
    stats['search_time'] = (end_time - start_time) * 1000
    stats['nodes_in_closed'] = len(closed_set)
    stats['nodes_in_open'] = open_set_size
    stats['memory_usage'] = len(g_score) + len(came_from) + len(closed_set)
    stats['avg_f_value'] = stats['total_f_sum'] / len(f_values) if f_values else 0
    stats['branching_factor'] = stats['nodes_visited'] / stats['nodes_expanded'] if stats['nodes_expanded'] > 0 else 0
    
    return [], stats


def print_stats(stats: Dict, title: str = "性能统计"):
    """打印性能统计信息"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f" 搜索时间: {stats['search_time']:.3f} ms")
    print(f" 扩展节点数: {stats['nodes_expanded']}")
    print(f" 访问节点数: {stats['nodes_visited']}")
    print(f" Open集合最大大小: {stats['nodes_in_open']}")
    print(f" Closed集合大小: {stats['nodes_in_closed']}")
    print(f" 内存使用(近似): {stats['memory_usage']} 个存储单元")
    print(f" 路径长度: {stats['path_length']}")
    print(f" 最优代价: {stats['optimal_cost']}")
    print(f" 解的深度: {stats['solution_depth']}")
    print(f" 平均分支因子: {stats['branching_factor']:.2f}")
    print(f" 有效分支因子: {stats['effective_branching']:.2f}")
    print(f" F值范围: [{stats['min_f_value']:.1f}, {stats['max_f_value']:.1f}]")
    print(f" 平均F值: {stats['avg_f_value']:.2f}")
    print(f"{'='*50}")


def compare_heuristics(start, end, grid):
    """比较不同启发函数的性能"""
    print("\n" + "="*60)
    print(" 启发函数性能对比")
    print("="*60)
    
    heuristics = ['manhattan', 'euclidean', 'zero', 'diagonal']
    results = {}
    
    for h in heuristics:
        path, stats = astar_maze_with_stats(start, end, grid, heuristic=h)
        results[h] = stats
        print(f"\n--- {h.upper()} 启发函数 ---")
        print_stats(stats, f"{h.upper()} 性能")
    
    # 对比表格
    print("\n" + "="*60)
    print(" 性能对比表")
    print("="*60)
    print(f"{'启发函数':<12} {'扩展节点':<10} {'访问节点':<10} {'时间(ms)':<10} {'路径长度':<10}")
    print("-"*60)
    for h in heuristics:
        s = results[h]
        print(f"{h:<12} {s['nodes_expanded']:<10} {s['nodes_visited']:<10} {s['search_time']:<10.3f} {s['path_length']:<10}")
    
    # 效率分析
    print("\n" + "="*60)
    print(" 效率分析")
    print("="*60)
    
    base = results['zero']['nodes_expanded']
    for h in ['manhattan', 'euclidean', 'diagonal']:
        reduction = (base - results[h]['nodes_expanded']) / base * 100
        print(f" {h} 相比 h=0 减少扩展节点: {reduction:.1f}%")
    
    optimal_cost = results['manhattan']['optimal_cost']
    print(f"\n 所有启发函数的最优代价: {optimal_cost}")
    for h in heuristics:
        if results[h]['optimal_cost'] != optimal_cost:
            print(f" {h} 未找到最优解")


def main():
    """主程序：处理任意输入"""
    print("请输入起点坐标 (row,col)，例如: (0,0)")
    start_input = input().strip()
    start = ast.literal_eval(start_input)
    
    print("请输入终点坐标 (row,col)，例如: (3,4)")
    end_input = input().strip()
    end = ast.literal_eval(end_input)
    
    print("请输入地图（二维数组），例如: [['S','.','.','X','.'],['.','X','.','.','.'],...]")
    print("提示：输入完成后先按回车，再按Ctrl+D（Linux/Mac）或Ctrl+Z（Windows），再按回车结束")
    grid_lines = []
    while True:
        try:
            line = input()
            grid_lines.append(line)
        except EOFError:
            break
    
    grid_str = ''.join(grid_lines).strip()
    
    try:
        grid = ast.literal_eval(grid_str)
    except:
        print("地图输入格式错误，请检查")
        return
    
    print("\n" + "="*60)
    print(" 使用曼哈顿距离的A*搜索")
    print("="*60)
    path, stats = astar_maze_with_stats(start, end, grid, heuristic='manhattan')
    
    print("\n搜索结果:")
    print(path)
    print_stats(stats, "曼哈顿距离性能")
    
    print("\n是否进行启发函数性能对比？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        compare_heuristics(start, end, grid)


if __name__ == "__main__":
    main()