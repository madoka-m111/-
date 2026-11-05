import heapq
import sys
import ast
import time
import re
from typing import List, Tuple, Dict

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """曼哈顿距离"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_weighted_with_stats(start: Tuple[int, int], 
                               end: Tuple[int, int], 
                               grid: List[List[str]],
                               heuristic='manhattan') -> Tuple[int, Dict]:
    """
    A*算法求解加权地形最小代价路径
    
    参数:
        heuristic: 'manhattan', 'manhattan_min', 'zero' 
    
    返回:
        (最小总代价, 统计信息)
    """
    rows, cols = len(grid), len(grid[0])
    
    # 统计信息
    stats = {
        'nodes_expanded': 0,      # 扩展节点数
        'nodes_visited': 0,       # 访问节点数
        'nodes_in_open': 0,       # open_set中最大节点数
        'nodes_in_closed': 0,     # closed_set中节点数
        'max_f_value': 0,         # 最大f值
        'min_f_value': float('inf'),  # 最小f值
        'avg_f_value': 0,         # 平均f值
        'total_f_sum': 0,         # f值总和
        'path_length': 0,         # 路径长度（步数）
        'total_cost': 0,          # 总代价
        'search_time': 0,         # 搜索时间
        'memory_usage': 0,        # 内存使用
        'branching_factor': 0,    # 平均分支因子
        'solution_depth': 0,      # 解的深度
        'effective_branching': 0, # 有效分支因子
        'optimal_cost': 0,        # 最优代价
        'path': [],               # 最优路径
    }
    
    # 获取进入格子的代价
    def get_cost(r: int, c: int) -> int:
        """获取进入该格子的代价"""
        if grid[r][c] == 'S':
            return 1
        elif grid[r][c] == 'E':
            return 0  # 终点代价为0
        else:
            return int(grid[r][c])
    
    # 计算地图中最小代价（用于改进启发函数）
    min_cost = float('inf')
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in ['S', 'E', 'X']:
                min_cost = min(min_cost, int(grid[r][c]))
    if min_cost == float('inf'):
        min_cost = 1
    
    # 选择启发函数
    def get_h(node):
        if heuristic == 'manhattan':
            # 标准曼哈顿距离（假设每步代价为1）
            return manhattan(node, end)
        elif heuristic == 'manhattan_min':
            # 改进：曼哈顿距离 × 最小代价
            return manhattan(node, end) * min_cost
        elif heuristic == 'zero':
            # h=0，退化为Dijkstra
            return 0
        elif heuristic == 'manhattan_weighted':
            # 加权曼哈顿距离（可能不可采纳）
            return manhattan(node, end) * 2
        else:
            return manhattan(node, end)
    
    start_time = time.time()
    
    # 优先队列: (f值, g值, 坐标)
    open_set = [(get_h(start), 0, start)]
    g_score = {start: 0}
    came_from = {}
    closed_set = set()
    
    # 用于统计的辅助数据结构
    f_values = []
    open_set_size = 1
    
    while open_set:
        # 更新open_set大小统计
        open_set_size = max(open_set_size, len(open_set))
        
        f, g, current = heapq.heappop(open_set)
        stats['nodes_expanded'] += 1
        
        # 更新f值统计
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
            path = path[::-1]
            
            # 计算统计信息
            end_time = time.time()
            stats['search_time'] = (end_time - start_time) * 1000  # 毫秒
            stats['path_length'] = len(path) - 1
            stats['total_cost'] = g_score[end]
            stats['nodes_in_closed'] = len(closed_set)
            stats['nodes_in_open'] = open_set_size
            stats['memory_usage'] = len(g_score) + len(came_from) + len(closed_set)
            stats['avg_f_value'] = stats['total_f_sum'] / len(f_values) if f_values else 0
            stats['branching_factor'] = stats['nodes_visited'] / stats['nodes_expanded'] if stats['nodes_expanded'] > 0 else 0
            stats['solution_depth'] = len(path) - 1
            stats['effective_branching'] = stats['nodes_expanded'] ** (1 / max(1, stats['solution_depth']))
            stats['optimal_cost'] = g_score[end]
            stats['path'] = path
            
            return g_score[end], stats
        
        closed_set.add(current)
        
        # 四个方向移动
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = current[0] + dr, current[1] + dc
            
            # 边界检查
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
                
            # 障碍物检查
            if grid[nr][nc] == 'X':
                continue
                
            neighbor = (nr, nc)
            move_cost = get_cost(nr, nc)
            tentative_g = g_score[current] + move_cost
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                h = get_h(neighbor)
                f = tentative_g + h
                heapq.heappush(open_set, (f, tentative_g, neighbor))
                came_from[neighbor] = current
                stats['nodes_visited'] += 1
    
    # 无路径情况
    end_time = time.time()
    stats['search_time'] = (end_time - start_time) * 1000
    stats['nodes_in_closed'] = len(closed_set)
    stats['nodes_in_open'] = open_set_size
    stats['memory_usage'] = len(g_score) + len(came_from) + len(closed_set)
    stats['avg_f_value'] = stats['total_f_sum'] / len(f_values) if f_values else 0
    stats['branching_factor'] = stats['nodes_visited'] / stats['nodes_expanded'] if stats['nodes_expanded'] > 0 else 0
    
    return -1, stats

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
    print(f" 总代价: {stats['total_cost']}")
    print(f" 最优代价: {stats['optimal_cost']}")
    print(f" 解的深度: {stats['solution_depth']}")
    print(f" 平均分支因子: {stats['branching_factor']:.2f}")
    print(f" 有效分支因子: {stats['effective_branching']:.2f}")
    print(f" F值范围: [{stats['min_f_value']:.1f}, {stats['max_f_value']:.1f}]")
    print(f" 平均F值: {stats['avg_f_value']:.2f}")
    if stats['path']:
        print(f" 最优路径: {stats['path']}")
    print(f"{'='*50}")

def print_path_details(path: List[Tuple[int, int]], grid: List[List[str]]):
    """打印路径详细代价信息"""
    if not path:
        print("无路径")
        return
    
    print("\n路径代价累计表:")
    print(f"{'步骤':<6} {'位置':<10} {'进入代价':<10} {'累计g值':<10}")
    print("-"*40)
    
    total_cost = 0
    for i, pos in enumerate(path):
        r, c = pos
        if grid[r][c] == 'S':
            cost = 0
        elif grid[r][c] == 'E':
            cost = 0
        else:
            cost = int(grid[r][c])
        
        if i > 0:  # 不是起点
            total_cost += cost
        
        print(f"{i:<6} {str(pos):<10} {cost:<10} {total_cost:<10}")
    
    print(f"\n总代价: {total_cost}")

def compare_heuristics(start: Tuple[int, int], end: Tuple[int, int], grid: List[List[str]]):
    """比较不同启发函数的性能"""
    print("\n" + "="*60)
    print(" 启发函数性能对比（加权地形）")
    print("="*60)
    
    heuristics = ['manhattan', 'manhattan_min', 'zero']
    results = {}
    
    for h in heuristics:
        cost, stats = astar_weighted_with_stats(start, end, grid, heuristic=h)
        results[h] = (cost, stats)
        print(f"\n--- {h.upper()} 启发函数 ---")
        print_stats(stats, f"{h.upper()} 性能")
    
    # 对比表格
    print("\n" + "="*60)
    print(" 性能对比表")
    print("="*60)
    print(f"{'启发函数':<15} {'扩展节点':<10} {'访问节点':<10} {'时间(ms)':<10} {'总代价':<10}")
    print("-"*60)
    for h in heuristics:
        cost, s = results[h]
        print(f"{h:<15} {s['nodes_expanded']:<10} {s['nodes_visited']:<10} {s['search_time']:<10.3f} {cost:<10}")
    
    # 效率分析
    print("\n" + "="*60)
    print(" 效率分析")
    print("="*60)
    
    # 计算相对效率
    base = results['zero'][1]['nodes_expanded']
    for h in ['manhattan', 'manhattan_min']:
        reduction = (base - results[h][1]['nodes_expanded']) / base * 100
        print(f" {h} 相比 h=0 减少扩展节点: {reduction:.1f}%")
    
    # 最优性验证
    optimal_cost = results['zero'][0]  # Dijkstra的结果作为基准
    print(f"\n 最优代价（Dijkstra基准）: {optimal_cost}")
    for h in heuristics:
        cost, _ = results[h]
        if cost == optimal_cost:
            print(f" ✓ {h} 找到最优解")
        else:
            print(f" ✗ {h} 未找到最优解 (找到{cost}, 最优{optimal_cost})")

def parse_input():
    """从标准输入解析任意格式的输入"""
    # 读取所有输入
    data = sys.stdin.read().strip()
    
    lines = data.split('\n')
        
    # 解析起点
    start = ast.literal_eval(lines[0].strip())
        
    # 解析终点
    end = ast.literal_eval(lines[1].strip())
        
    # 解析地图（可能跨多行）
    grid_str = '\n'.join(lines[2:]).strip()
    try:
        # 自动给裸字母 S/E/X 加双引号（简单正则替换）
        import re
        # 将独立的大写字母替换为带引号的格式
        grid_str = re.sub(r'(?<![a-zA-Z0-9"\'_])([A-Z])(?![a-zA-Z0-9"\'_])', r'"\1"', grid_str)
        
        grid = ast.literal_eval(grid_str)
    except Exception as e:
        print("地图输入格式错误，请检查")
        print(f"尝试解析的内容: {grid_str}")
        return
        
    return start, end, grid

def main():
    """主程序：处理任意输入"""
    # 读取输入
    print("请输入起点坐标 (row,col)，例如: (0,0)")
    start_input = input().strip()
    start = ast.literal_eval(start_input)
    
    print("请输入终点坐标 (row,col)，例如: (4,4)")
    end_input = input().strip()
    end = ast.literal_eval(end_input)
    
    print("请输入地图（二维数组），例如: [['S','1','1','8','1'],['1','8','3','1','1'],...]")
    print("提示：输入完成后先按回车，再按Ctrl+D（Linux/Mac）或Ctrl+Z（Windows），再按回车结束")
    grid_lines = []
    while True:
        try:
            line = input()
            grid_lines.append(line)
        except EOFError:
            break
    
    # 合并地图输入（保留换行符）
    grid_str = '\n'.join(grid_lines).strip()
    
    # 解析地图 - 使用与parse_input相同的正则处理
    try:
        # 自动给裸字母 S/E/X 加双引号
        grid_str = re.sub(r'(?<![a-zA-Z0-9"\'_])([A-Z])(?![a-zA-Z0-9"\'_])', r'"\1"', grid_str)
        
        grid = ast.literal_eval(grid_str)
    except Exception as e:
        print("地图输入格式错误，请检查")
        print(f"尝试解析的内容: {grid_str}")
        print(f"错误详情: {e}")
        return
    
    # 运行A*算法（使用曼哈顿距离）
    print("\n" + "="*60)
    print(" 使用曼哈顿距离的A*搜索（加权地形）")
    print("="*60)
    cost, stats = astar_weighted_with_stats(start, end, grid, heuristic='manhattan')
    
    print(f"\n最小总代价: {cost}")
    print_stats(stats, "曼哈顿距离性能")
    
    # 打印路径详细代价
    if stats['path']:
        print_path_details(stats['path'], grid)
    
    # 对比不同启发函数
    print("\n是否进行启发函数性能对比？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        compare_heuristics(start, end, grid)

if __name__ == "__main__":
    main()