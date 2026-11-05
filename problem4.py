import sys
import json
import time

# ========== A* 最短路径 ==========
def a_star_shortest_path(start, goal, game_map):
    rows, cols = len(game_map), len(game_map[0])
    start, goal = tuple(start), tuple(goal)
    
    def heuristic(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    open_set = [start]
    came_from = {}
    
    while open_set:
        open_set.sort(key=lambda x: f_score.get(x, float('inf')))
        current = open_set.pop(0)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return g_score[goal], path
        
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and game_map[nx][ny] != 'X':
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    if neighbor not in open_set:
                        open_set.append(neighbor)
    return float('inf'), []

# ========== 距离矩阵 ==========
def compute_distance_matrix(game_map, start_a, start_b, treasures):
    start_a, start_b = tuple(start_a), tuple(start_b)
    points = {'A': start_a, 'B': start_b}
    for name, info in treasures.items():
        points[name] = tuple(info['pos'])
    
    dist = {}
    keys = list(points.keys())
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j:
                dist[(keys[i], keys[j])] = 0
            else:
                d, _ = a_star_shortest_path(points[keys[i]], points[keys[j]], game_map)
                dist[(keys[i], keys[j])] = d
    return dist, points

# ========== 博弈树节点 ==========
class Node:
    def __init__(self, player, remaining, a_pos, b_pos, a_val, b_val, depth, node_id):
        self.player = player          # 'A' 或 'B'
        self.remaining = remaining    # 剩余宝藏集合
        self.a_pos = a_pos            # A当前位置
        self.b_pos = b_pos            # B当前位置
        self.a_val = a_val            # A已获得价值
        self.b_val = b_val            # B已获得价值
        self.depth = depth            # 当前深度
        self.node_id = node_id        # 节点编号
        self.children = []            # (宝藏名, 子节点) 列表
        self.value = None             # 节点评估值
        self.best_move = None         # 最佳走法

# ========== 构建博弈树 ==========
def build_game_tree(dist, depth, player, remaining, a_pos, b_pos, a_val, b_val, treasure_values, node_id=0):
    node = Node(player, remaining, a_pos, b_pos, a_val, b_val, depth, node_id)
    
    # 达到最大深度或没有剩余宝藏
    if depth == 0 or not remaining:
        node.value = a_val - b_val
        return node
    
    targets = sorted(list(remaining))
    next_player = 'B' if player == 'A' else 'A'
    child_id = node_id + 1
    
    for t in targets:
        if player == 'A':
            # A选择去宝藏t
            d = dist[(a_pos, t)]
            new_a_val = a_val + treasure_values[t] - d  # 扣除移动代价
            new_a_pos = t
            child = build_game_tree(dist, depth-1, next_player, remaining - {t}, 
                                   new_a_pos, b_pos, new_a_val, b_val, treasure_values, child_id)
        else:
            # B选择去宝藏t
            d = dist[(b_pos, t)]
            new_b_val = b_val + treasure_values[t] - d  # 扣除移动代价
            new_b_pos = t
            child = build_game_tree(dist, depth-1, next_player, remaining - {t},
                                   a_pos, new_b_pos, a_val, new_b_val, treasure_values, child_id)
        
        node.children.append((t, child))
        child_id += 1
    
    return node

# ========== Alpha-Beta 剪枝 ==========
def alpha_beta(node, alpha, beta, stats):
    stats['nodes_visited'] += 1
    
    # 叶子节点
    if not node.children:
        node.value = node.a_val - node.b_val
        stats['leaf_nodes_visited'] += 1
        return node.value
    
    if node.player == 'A':  # MAX节点
        val = float('-inf')
        for t, ch in node.children:
            v = alpha_beta(ch, alpha, beta, stats)
            if v > val:
                val = v
                node.best_move = t
            alpha = max(alpha, val)
            if alpha >= beta:
                stats['pruned_nodes'] += 1
                break
        node.value = val
        return val
    else:  # MIN节点
        val = float('inf')
        for t, ch in node.children:
            v = alpha_beta(ch, alpha, beta, stats)
            if v < val:
                val = v
                node.best_move = t
            beta = min(beta, val)
            if alpha >= beta:
                stats['pruned_nodes'] += 1
                break
        node.value = val
        return val

# ========== 普通 Minimax ==========
def minimax(node, stats):
    stats['nodes_visited'] += 1
    
    if not node.children:
        node.value = node.a_val - node.b_val
        stats['leaf_nodes_visited'] += 1
        return node.value
    
    if node.player == 'A':  # MAX节点
        val = float('-inf')
        for t, ch in node.children:
            v = minimax(ch, stats)
            if v > val:
                val = v
                node.best_move = t
        node.value = val
        return val
    else:  # MIN节点
        val = float('inf')
        for t, ch in node.children:
            v = minimax(ch, stats)
            if v < val:
                val = v
                node.best_move = t
        node.value = val
        return val

# ========== 寻找最佳首步 ==========
def find_best_move(root):
    """从根节点找到最佳首步"""
    if not root.children:
        return None, root.value
    
    # 根节点是A（MAX），选择价值最大的子节点
    best_val = float('-inf')
    best_t = None
    for t, ch in root.children:
        if ch.value > best_val:
            best_val = ch.value
            best_t = t
    
    return best_t, best_val

# ========== 统计与打印 ==========
def count_nodes(node):
    cnt = 1
    for _, ch in node.children:
        cnt += count_nodes(ch)
    return cnt

def print_distance_matrix(dist, points):
    keys = sorted(list(points.keys()))
    print("\n距离矩阵:")
    print("    " + "  ".join(f"{k:>4}" for k in keys))
    for k1 in keys:
        row = [f"{k1:>4}"]
        for k2 in keys:
            val = dist[(k1,k2)]
            if val == float('inf'):
                row.append("  ∞")
            else:
                row.append(f"{val:>4}")
        print(" ".join(row))

def print_stats(stats, title):
    print(f"\n{title}")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  访问节点数: {stats['nodes_visited']}")
    print(f"  叶子节点访问: {stats['leaf_nodes_visited']}")
    print(f"  剪枝节点数: {stats['pruned_nodes']}")
    print(f"  搜索时间: {stats['search_time']:.3f} ms")
    if stats['total_nodes'] > 0:
        print(f"  剪枝率: {stats['pruned_nodes']/stats['total_nodes']*100:.1f}%")

def compare_algorithms(root, depth, title):
    """真实对比Minimax和Alpha-Beta"""
    print(f"\n{'='*60}")
    print(f" Minimax vs Alpha-Beta 对比 {title}")
    print(f"{'='*60}")
    
    # 重新构建树用于Minimax（避免相互影响）
    root_mm = build_game_tree(
        root.__dict__.get('dist', {}),  # 需要传递dist，这里简化处理
        depth, 'A', root.remaining, root.a_pos, root.b_pos, 
        root.a_val, root.b_val, 
        {t: 0 for t in root.remaining}  # 简化：值不影响对比
    )
    
    # 实际上需要重新构建完整的树，这里用原始数据
    # 由于函数签名问题，我们直接在主函数中构建两棵树
    
    # 这里仅打印对比信息，实际对比在主函数中完成
    print("  (对比结果见上方统计信息)")

# ========== 主函数 ==========
def main():
    print("=" * 60)
    print(" 题目四：地图寻宝博弈 (Minimax vs Alpha-Beta)")
    print("=" * 60)
    print("\n请输入地图和宝藏数据（JSON格式）：")
    print("格式: {'map': [...], 'treasures': {...}, 'start_a': [x,y], 'start_b': [x,y]}")
    print("提示：输入完成后先按回车，再按Ctrl+D（Linux/Mac）或Ctrl+Z（Windows），再按回车结束")
    
    data = sys.stdin.read().strip()
    if not data:
        print("\n(未检测到输入，自动加载默认测试用例...)")
        data = '''{
            "map": [
                ["A","1","1","X","2","1","T1"],
                ["1","X","6","X","1","X","1"],
                ["1","1","T2","1","1","1","1"],
                ["X","1","X","8","X","1","X"],
                ["1","1","1","1","T3","1","1"],
                ["1","X","9","X","1","X","1"],
                ["T4","1","1","X","1","1","B"]
            ],
            "treasures": {
                "T1":{"pos":[0,6],"value":9},
                "T2":{"pos":[2,2],"value":6},
                "T3":{"pos":[4,4],"value":7},
                "T4":{"pos":[6,0],"value":7}
            },
            "start_a":[0,0],
            "start_b":[6,6]
        }'''
    
    try:
        input_data = json.loads(data)
        input_data['start_a'] = tuple(input_data['start_a'])
        input_data['start_b'] = tuple(input_data['start_b'])
        for name, info in input_data['treasures'].items():
            info['pos'] = tuple(info['pos'])
    except Exception as e:
        print(f"JSON解析错误: {e}")
        return
    
    game_map = input_data['map']
    treasures = input_data['treasures']
    start_a = input_data.get('start_a', (0,0))
    start_b = input_data.get('start_b', (6,6))
    
    treasure_names = list(treasures.keys())
    treasure_values = {n: treasures[n]['value'] for n in treasure_names}
    
    print("\n" + "="*60)
    print(" 第一步：计算距离矩阵")
    print("="*60)
    dist, points = compute_distance_matrix(game_map, start_a, start_b, treasures)
    print_distance_matrix(dist, points)
    
    for depth in [2, 4]:
        print(f"\n{'='*60}")
        print(f" 第二步：求解 H={depth}")
        print(f"{'='*60}")
        
        # ===== Alpha-Beta =====
        root_ab = build_game_tree(dist, depth, 'A', set(treasure_names), 'A', 'B', 0, 0, treasure_values, node_id=0)
        stats_ab = {
            'nodes_visited': 0,
            'leaf_nodes_visited': 0,
            'pruned_nodes': 0,
            'search_time': 0,
            'total_nodes': count_nodes(root_ab)
        }
        
        t0 = time.time()
        alpha_beta(root_ab, float('-inf'), float('inf'), stats_ab)
        stats_ab['search_time'] = (time.time() - t0) * 1000
        
        best_target_ab, best_value_ab = find_best_move(root_ab)
        output_ab = {"best_target": best_target_ab, "best_value": best_value_ab}
        
        # ===== Minimax =====
        root_mm = build_game_tree(dist, depth, 'A', set(treasure_names), 'A', 'B', 0, 0, treasure_values, node_id=0)
        stats_mm = {
            'nodes_visited': 0,
            'leaf_nodes_visited': 0,
            'pruned_nodes': 0,
            'search_time': 0,
            'total_nodes': count_nodes(root_mm)
        }
        
        t0 = time.time()
        minimax(root_mm, stats_mm)
        stats_mm['search_time'] = (time.time() - t0) * 1000
        
        best_target_mm, best_value_mm = find_best_move(root_mm)
        output_mm = {"best_target": best_target_mm, "best_value": best_value_mm}
        
        # ===== 打印结果 =====
        print(f"\nAlpha-Beta 结果: {json.dumps(output_ab, ensure_ascii=False)}")
        print(f"Minimax    结果: {json.dumps(output_mm, ensure_ascii=False)}")
        
        # ===== 打印统计信息 =====
        print_stats(stats_ab, f"\nAlpha-Beta 统计 (H={depth})")
        print_stats(stats_mm, f"\nMinimax 统计 (H={depth})")
        
        # ===== 对比分析 =====
        print(f"\n{'='*60}")
        print(f" 对比分析 (H={depth})")
        print(f"{'='*60}")
        print(f"{'指标':<20} {'Alpha-Beta':<15} {'Minimax':<15}")
        print("-"*50)
        print(f"{'最佳目标':<20} {best_target_ab:<15} {best_target_mm:<15}")
        print(f"{'最佳价值':<20} {best_value_ab:<15} {best_value_mm:<15}")
        print(f"{'访问节点数':<20} {stats_ab['nodes_visited']:<15} {stats_mm['nodes_visited']:<15}")
        print(f"{'叶子节点访问':<20} {stats_ab['leaf_nodes_visited']:<15} {stats_mm['leaf_nodes_visited']:<15}")
        print(f"{'剪枝节点数':<20} {stats_ab['pruned_nodes']:<15} {stats_mm['pruned_nodes']:<15}")
        print(f"{'搜索时间(ms)':<20} {stats_ab['search_time']:<15.3f} {stats_mm['search_time']:<15.3f}")
        
        if stats_mm['nodes_visited'] > 0:
            reduction = (stats_mm['nodes_visited'] - stats_ab['nodes_visited']) / stats_mm['nodes_visited'] * 100
            print(f"\n 节点访问减少率: {reduction:.1f}%")
        
        print(f"\n 结果一致性: {'一致' if output_ab == output_mm else '不一致'}")
        print(f"{'='*60}")
    
    print("\n" + "="*60)
    print(" 题目四完成！")
    print("="*60)

if __name__ == "__main__":
    main()