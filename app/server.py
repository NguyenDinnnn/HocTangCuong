from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
from threading import Lock
import os, pickle, torch, numpy as np, time
import random
import heapq
from itertools import permutations
from collections import defaultdict
import torch.nn.functional as F

from app.robot_env import GridWorldEnv
# Lưu ý: Giả định thư viện clients đã được thêm vào PYTHONPATH hoặc thư mục cha
# Nếu không, cần đảm bảo robot_env và ActorCritic có thể được import
from clients.train_a2c import ActorCritic

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="RL Robot API", version="1.0.0")
_env_lock = Lock()
# Mount thư mục web
app.mount("/web", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../clients/web")), name="web")

# ---------------------------
# State Abstraction Functions (Đồng bộ với qlearning_trainer_optimized.py)
# ---------------------------

def encode_visited(wp_list, visited_set):
    """Mã hóa trạng thái Waypoint đã thăm dưới dạng một số nguyên."""
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

def get_abstract_state(env):
    """
    Trả về trạng thái trừu tượng hóa (relative direction) cho phép tổng quát hóa.
    Keys: (sign_dx_goal, sign_dy_goal, sign_dx_nearest_wp, sign_dy_nearest_wp, 
            visited_code, local_obstacle_code)
    
    Sign function: -1 (phải đi Lùi/Xuống), 0 (đã ngang hàng), 1 (phải đi Tới/Lên).
    """
    robot_x, robot_y = env.get_state()
    goal_x, goal_y = env.goal
    waypoints = env.waypoints
    obstacles = env.obstacles
    visited_waypoints = env.visited_waypoints
    
    # Hàm Sign: Lấy hướng tương đối
    def get_sign(delta):
        if delta > 0: return 1
        if delta < 0: return -1
        return 0
    
    # 1. Delta tới Goal (dưới dạng Sign)
    delta_x_goal = goal_x - robot_x
    delta_y_goal = goal_y - robot_y
    sign_dx_goal = get_sign(delta_x_goal)
    sign_dy_goal = get_sign(delta_y_goal)
    
    # 2. Delta tới Waypoint gần nhất (chưa thăm)
    targets = [wp for wp in waypoints if wp not in visited_waypoints]
    
    sign_dx_nearest_wp = 0
    sign_dy_nearest_wp = 0
    
    if targets:
        # Tìm Waypoint gần nhất (dùng khoảng cách Manhattan)
        def manhattan_distance(wp):
            return abs(wp[0]-robot_x) + abs(wp[1]-robot_y)
            
        nearest_wp = min(targets, key=manhattan_distance)
        
        delta_x_nearest_wp = nearest_wp[0] - robot_x
        delta_y_nearest_wp = nearest_wp[1] - robot_y
        
        sign_dx_nearest_wp = get_sign(delta_x_nearest_wp)
        sign_dy_nearest_wp = get_sign(delta_y_nearest_wp)
    
    # 3. Visited Code
    visited_code = encode_visited(waypoints, visited_waypoints)
    
    # 4. Local Obstacle Code (4-bit code)
    # Lên (0b0001), Phải (0b0010), Xuống (0b0100), Trái (0b1000)
    local_obstacle_code = 0
    
    current_width = env.width
    current_height = env.height
    
    neighbors = {
        'up': (robot_x, robot_y - 1),
        'right': (robot_x + 1, robot_y),
        'down': (robot_x, robot_y + 1),
        'left': (robot_x - 1, robot_y),
    }

    # Kiểm tra xem ô lân cận có phải là chướng ngại vật hay tường không
    if neighbors['up'] in obstacles or not (0 <= neighbors['up'][1] < current_height):
        local_obstacle_code |= 0b0001
    if neighbors['right'] in obstacles or not (0 <= neighbors['right'][0] < current_width):
        local_obstacle_code |= 0b0010
    if neighbors['down'] in obstacles or not (0 <= neighbors['down'][1] < current_height):
        local_obstacle_code |= 0b0100
    if neighbors['left'] in obstacles or not (0 <= neighbors['left'][0] < current_width):
        local_obstacle_code |= 0b1000
    
    # Kết hợp các đặc trưng tương đối thành Full Abstract State Key (6 đặc trưng)
    return (sign_dx_goal, sign_dy_goal, 
            sign_dx_nearest_wp, sign_dy_nearest_wp, 
            visited_code, local_obstacle_code)

# ---------------------------
# Environment
# ---------------------------
GRID_W, GRID_H = 10, 8
START = (0, 0)
GOAL = (9, 7)
WAYPOINTS = [(3, 2), (6, 5)]
OBSTACLES = [(1, 1), (2, 3), (4, 4), (5, 1), (7, 6)]
num_waypoints = 2  # số waypoint mặc định
def _new_qrow():
    return {a: 0.0 for a in ['up', 'right', 'down', 'left']}
epsilon_decay_rate = 0.995
epsilon_min = 0.1
epsilon_decay = 0.995
# CẬP NHẬT: Đặt giá trị mặc định cho max_steps
env = GridWorldEnv(GRID_W, GRID_H, START, GOAL, OBSTACLES, WAYPOINTS, max_steps=500)

REWARD_PARAMS = {
    "step_penalty": -0.01,     # Very small step penalty
    "wall_penalty": -1,        # Moderate wall penalty  
    "obstacle_penalty": -2,    # Moderate obstacle penalty
    "waypoint_reward": 10,     # Reasonable waypoint reward
    "goal_reward": 50,         # High goal reward
    "goal_before_waypoints_penalty": -10,  # Penalty for wrong goal timing
}

DEFAULT_MAX_STEPS = 500

env = GridWorldEnv(GRID_W, GRID_H, START, GOAL, OBSTACLES, WAYPOINTS, max_steps=DEFAULT_MAX_STEPS)
# gán reward params vào env
for k, v in REWARD_PARAMS.items():
    setattr(env, k, v)

# ---------------------------
# Models dir
# ---------------------------
models_dir = os.path.join(os.path.dirname(__file__), "../clients/models")
os.makedirs(models_dir, exist_ok=True)

# ---------------------------
# Load MC
# ---------------------------


mc_qfile = os.path.join(models_dir, "mc_qtable.pkl")
if os.path.exists(mc_qfile):
    with open(mc_qfile, "rb") as f:
        loaded_mc_Q = pickle.load(f)
    # CẬP NHẬT: Khôi phục defaultdict và cập nhật dữ liệu
    mc_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    mc_Q.update(loaded_mc_Q)
else:
    mc_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
      

# ---------------------------
# Load Q-learning
# ---------------------------
ql_qfile = os.path.join(models_dir, "qlearning_qtable.pkl")
if os.path.exists(ql_qfile):
    with open(ql_qfile, "rb") as f:
        loaded_ql_Q = pickle.load(f)
    # CẬP NHẬT: Khôi phục defaultdict và cập nhật dữ liệu
    ql_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    ql_Q.update(loaded_ql_Q)
else:
    ql_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})

# ---------------------------
## Load SARSA
# ---------------------------
sarsa_qfile = os.path.join(models_dir, "sarsa_qtable.pkl")
if os.path.exists(sarsa_qfile):
    with open(sarsa_qfile, "rb") as f:
        loaded_sarsa_Q = pickle.load(f)
    # Khôi phục defaultdict và cập nhật dữ liệu
    sarsa_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    sarsa_Q.update(loaded_sarsa_Q)
else:
    sarsa_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    
# ---------------------------
# Load A2C
# ---------------------------
a2c_model_file = os.path.join(models_dir, "a2c_model.pth")
in_channels = 5
n_actions = len(env.ACTIONS)
a2c_model = ActorCritic(in_channels, env.height, env.width, n_actions)
if os.path.exists(a2c_model_file):
    try:
        # Load model trên CPU nếu không có GPU
        a2c_model.load_state_dict(torch.load(a2c_model_file, map_location=torch.device('cpu')))
        a2c_model.eval()
        print("✅ A2C model loaded successfully")
    except Exception as e:
        print(f"⚠️ Không load được A2C checkpoint: {e}. Dùng model mới.")

# ---------------------------
# Load Q-learning (Đã cập nhật để dùng file abstract TỐI ƯU)
# ---------------------------
# ĐỒNG BỘ: Sử dụng tên file mô hình Q-learning đã được huấn luyện với state abstraction TỐI ƯU.
ql_qfile = os.path.join(models_dir, "qlearning_qtable_abstract_optimized.pkl") 
if os.path.exists(ql_qfile):
    with open(ql_qfile, "rb") as f:
        loaded_ql_Q = pickle.load(f)
    # Khôi phục defaultdict
    ql_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    # Kiểm tra và tải Q-table từ định dạng mới (có chứa 'Q' và 'epsilon')
    if isinstance(loaded_ql_Q, dict) and 'Q' in loaded_ql_Q:
        ql_Q.update(loaded_ql_Q['Q'])
        # Tải Epsilon (nếu có, để đảm bảo tính nhất quán khi chạy trên Server)
        epsilon = loaded_ql_Q.get('epsilon', 1.0) 
    else: # Hỗ trợ định dạng cũ (chỉ Q-table)
        ql_Q.update(loaded_ql_Q)
        epsilon = 1.0 # Đặt lại epsilon nếu load định dạng cũ
    print(f"✅ Đã load Q-learning model từ: {ql_qfile}")
else:
    ql_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    print(f"⚠️ Không tìm thấy Q-learning model tại: {ql_qfile}. Sử dụng Q-table rỗng.")

# ---------------------------
# RL params
# ---------------------------
actions = ['up', 'right', 'down', 'left']
alpha, gamma = 0.1, 0.99
epsilon = 1.0
mc_Q = defaultdict(lambda: {a: 0.1 for a in actions})
mc_N = defaultdict(lambda: {a: 0 for a in actions})
# ---------------------------
# Request Models
# ---------------------------
class ResetRequest(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None
    waypoints: Optional[List[Tuple[int, int]]] = None
    obstacles: Optional[List[Tuple[int, int]]] = None
    max_steps: Optional[int] = None

class ActionInput(BaseModel):
    action: Optional[int] = None
    action_name: Optional[str] = None

class AlgorithmRequest(BaseModel):
    algorithm: str

class AStarRequest(BaseModel):
    goal: Optional[Tuple[int, int]] = None

class SarsaTrainRequest(BaseModel):
    episodes: int = 2000

def encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

def a_star(start, goal, obstacles, width, height):
    open_set = []
    heapq.heappush(open_set, (0+abs(start[0]-goal[0])+abs(start[1]-goal[1]), 0, start, [start]))
    visited = set()
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
        
        if current in visited:
            continue
        visited.add(current)
        x,y = current
        for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<width and 0<=ny<height and (nx,ny) not in obstacles:
                heapq.heappush(open_set, (g+1+abs(nx-goal[0])+abs(ny-goal[1]), g+1, (nx,ny), path+[(nx,ny)]))
    return []

def plan_path_through_waypoints(start, waypoints, goal, obstacles, width, height):
    best_path = None
    min_len = float('inf')
    for order in permutations(waypoints):
        path = []
        curr = start
        valid = True
        for wp in order:
            sub_path = a_star(curr, wp, obstacles, width, height)
            if not sub_path:
                valid = False
                break
            path += sub_path[:-1]
            curr = wp
        if not valid:
            continue
        sub_path = a_star(curr, goal, obstacles, width, height)
        if not sub_path:
            continue
        path += sub_path
        if len(path) < min_len:
            min_len = len(path)
            best_path = path
    return best_path or []

# ---------------------------
# Simple reward shaping for better Q-Learning
# ---------------------------
def _apply_simple_reward_shaping(old_state, new_state, base_reward):
    """Simple distance-based reward shaping"""
    old_x, old_y = old_state
    new_x, new_y = new_state
    goal_x, goal_y = env.goal
    
    old_distance = abs(old_x - goal_x) + abs(old_y - goal_y)
    new_distance = abs(new_x - goal_x) + abs(new_y - goal_y)
    
    # Simple reward shaping: +0.1 for getting closer, -0.1 for moving away
    if new_distance < old_distance:
        return base_reward + 0.1
    elif new_distance > old_distance:
        return base_reward - 0.1
    else:
        return base_reward

def _apply_goal_guard_and_unified_reward(r: float, reached_goal_now: bool) -> float:
    if reached_goal_now and len(env.visited_waypoints) < len(env.waypoints):
        r += REWARD_PARAMS["goal_before_waypoints_penalty"]
    return r

# ---------------------------
# Extend GridWorldEnv for A* step (RL-style reward)
# ---------------------------
def step_to_rl(self, target):
    """
    Di chuyển robot đến ô target, tính toán reward theo cách RL 
    (thu thập waypoint, về đích).
    """
    self.state = target
    self.steps += 1
    reward = -0.1 # Penalty cho mỗi bước đi
    done = False
    
    if target in self.waypoints and target not in self.visited_waypoints:
        self.visited_waypoints.add(target)
        reward = 1.0 # Reward khi ghé thăm Waypoint
        
    # Điều kiện hoàn thành: Đã đến Goal VÀ đã ghé thăm tất cả Waypoint
    if target == self.goal and len(self.visited_waypoints) == len(self.waypoints):
        done = True
        reward = 10.0 # Reward lớn khi hoàn thành
        
    info = {"note": "Auto move by A* (RL reward)"}
    return target, reward, done, info

GridWorldEnv.step_to = step_to_rl
# ---------------------------
# Endpoints
# ---------------------------
@app.get("/map")
def get_map():
    with _env_lock:
        return {"map": env.get_map()}

@app.post("/reset")
def reset(req: ResetRequest):
    global env
    with _env_lock:
        w = req.width or env.width
        h = req.height or env.height
        s = req.start or env.start
        g = req.goal or env.goal
        wp = req.waypoints if req.waypoints is not None else list(env.waypoints)
        ob = req.obstacles if req.obstacles is not None else list(env.obstacles)
        ms = req.max_steps if req.max_steps is not None else 500

        env = GridWorldEnv(w, h, s, g, ob, wp, max_steps=ms)
        env.episode_buffer = []
        env.state_visit_count = defaultdict(int)
        state = env.reset(max_steps=ms)
        
        return {"state": state, "map": env.get_map(), "ascii": env.render_ascii()}

@app.post("/reset_all")
def reset_all():
    """Tạo môi trường mới với Waypoint, Goal và Obstacle ngẫu nhiên."""
    global env, epsilon
    with _env_lock:
        w, h = env.width, env.height
        start = (0, 0)

        # Random obstacles
        all_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) != start]
        random.shuffle(all_cells)
        obstacles = all_cells[:8]   # ví dụ chọn 8 chướng ngại vật

        # Random 2 waypoint + 1 goal
        remain = [cell for cell in all_cells if cell not in obstacles]
        wp_goal_candidate = remain
        
        # Đảm bảo có đủ ô cho Waypoint và Goal
        if len(wp_goal_candidate) < num_waypoints + 1:
            return {"error": "Không đủ ô trống để tạo waypoint và goal ngẫu nhiên."}
            
        waypoints = wp_goal_candidate[:num_waypoints]
        goal = wp_goal_candidate[num_waypoints]

        env = GridWorldEnv(w, h, start, goal, obstacles, waypoints, max_steps=500)
        state = env.reset(max_steps=500)

        return {
            "state": state,
            "map": env.get_map(),
            "ascii": env.render_ascii(),
            "obstacles": obstacles,
            "waypoints": waypoints,
            "goal": goal,
            "rewards_over_time": []   # reset luôn biểu đồ
        }

@app.get("/state")
def get_state():
    with _env_lock:
        return {
            "state": env.get_state(),
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "ascii": env.render_ascii()
        }

@app.post("/step")
def step(inp: ActionInput):
    """Thực hiện một bước đi thủ công."""
    with _env_lock:
        try:
            if inp.action_name is not None:
                s,r,done,info = env.step_by_name(inp.action_name)
            elif inp.action is not None:
                s,r,done,info = env.step(inp.action)
            else:
                return {"error": "No action provided"}
            return {
                "state": s,
                "reward": r,
                "done": done,
                "info": info,
                "steps": env.steps,
                "visited_waypoints": list(env.visited_waypoints),
                "ascii": env.render_ascii()
            }
        except ValueError as e:
            return {"error": str(e)}

# ---------------------------
# RL step-by-step (học / lưu / load mỗi bước)
# ---------------------------
def _encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

def _save_and_reload_q(qtable, path):
    with open(path, "wb") as f:
        pickle.dump(qtable, f)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    d = defaultdict(_new_qrow)
    d.update(loaded)
    return d

def _save_and_reload_a2c(model, path):
    torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path))
    model.eval()

def _select_action_from(Q, st, eps):
    # Đảm bảo state có entry trong Q-table
    if st not in Q:
        Q[st] = _new_qrow()
    
    # Simple epsilon-greedy
    if np.random.rand() < eps:
        # Exploration with slight bias towards goal
        x, y = st
        goal_x, goal_y = env.goal
        
        # 60% random, 40% biased towards goal
        if np.random.rand() < 0.4:
            best_action = 'right'  # default
            best_distance = float('inf')
            
            for action in actions:
                action_idx = actions.index(action)
                dx, dy = env.ACTIONS[action_idx]
                new_x, new_y = x + dx, y + dy
                
                if (0 <= new_x < env.width and 0 <= new_y < env.height and 
                    (new_x, new_y) not in env.obstacles):
                    distance = abs(new_x - goal_x) + abs(new_y - goal_y)
                    if distance < best_distance:
                        best_distance = distance
                        best_action = action
            return best_action
        else:
            return np.random.choice(actions)
    else:
        # Exploitation: choose best Q-value
        return max(Q[st], key=Q[st].get)

def _heuristic_guided_exploration(state):
    """Select exploration action biased towards goal direction"""
    x, y = state
    goal_x, goal_y = env.goal
    
    # Calculate heuristic scores for each action
    action_scores = {}
    for action in actions:
        action_idx = actions.index(action)
        dx, dy = env.ACTIONS[action_idx]
        new_x, new_y = x + dx, y + dy
        
        # Check if action is valid (not hitting walls or obstacles)
        if (0 <= new_x < env.width and 0 <= new_y < env.height and 
            (new_x, new_y) not in env.obstacles):
            # Manhattan distance to goal (lower is better)
            distance = abs(new_x - goal_x) + abs(new_y - goal_y)
            action_scores[action] = 1.0 / (distance + 1)  # Convert to score (higher is better)
        else:
            action_scores[action] = 0.0  # Invalid action gets lowest score
    
    # Select action with probability proportional to heuristic scores
    if sum(action_scores.values()) > 0:
        total_score = sum(action_scores.values())
        probabilities = [action_scores[action] / total_score for action in actions]
        return np.random.choice(actions, p=probabilities)
    else:
        return np.random.choice(actions)  # Fallback to random if all actions are invalid

def _select_best_heuristic_action(state, candidate_actions):
    """Select action from candidates using heuristic"""
    x, y = state
    goal_x, goal_y = env.goal
    
    best_action = candidate_actions[0]
    best_distance = float('inf')
    
    for action in candidate_actions:
        action_idx = actions.index(action)
        dx, dy = env.ACTIONS[action_idx]
        new_x, new_y = x + dx, y + dy
        
        # Check if action leads to valid position
        if (0 <= new_x < env.width and 0 <= new_y < env.height and 
            (new_x, new_y) not in env.obstacles):
            # Calculate Manhattan distance to goal
            distance = abs(new_x - goal_x) + abs(new_y - goal_y)
            if distance < best_distance:
                best_distance = distance
                best_action = action
    
    return best_action

@app.post("/step_algorithm")
def step_algorithm(req: AlgorithmRequest):
    global epsilon
    algo = req.algorithm
    with _env_lock:
        state_xy = env.get_state()
        done = False
        reward = 0

        def encode_visited(wp_list, visited_set):
            code = 0
            for i, wp in enumerate(wp_list):
                if wp in visited_set:
                    code |= (1 << i)
            return code
        
        # Q-learning sử dụng Abstract State (Relative Directions, Visited_Code, Obstacle Code)
        abstract_state = get_abstract_state(env)
        
        visited_code = encode_visited(env.waypoints, env.visited_waypoints)
        full_state = (state_xy[0], state_xy[1], visited_code)

        if algo == "MC":
             # --- Khởi tạo episode_buffer nếu chưa có ---
            if not hasattr(env, 'episode_buffer'):
                env.episode_buffer = []

            # --- Mã hóa trạng thái ---
            def encode_visited(wp_list, visited_set):
                code = 0
                for i, wp in enumerate(wp_list):
                    if wp in visited_set:
                        code |= (1 << i)
                return code

            state_xy = env.get_state()
            visited_code = encode_visited(env.waypoints, env.visited_waypoints)
            full_state = (state_xy[0], state_xy[1], visited_code)
            # --- Chọn hành động theo epsilon-greedy ---
            if np.random.rand() > epsilon:
                if full_state in mc_Q and any(mc_Q[full_state].values()):
                    action_name = max(mc_Q[full_state], key=mc_Q[full_state].get)
                else:
                    # Hướng tới waypoint chưa thăm, nếu hết → goal
                    targets = [wp for wp in env.waypoints if wp not in env.visited_waypoints] or [env.goal]
                    target = targets[0]
                    dx, dy = target[0]-state_xy[0], target[1]-state_xy[1]
                    if abs(dx) > abs(dy):
                        action_name = 'right' if dx>0 else 'left'
                    else:
                        action_name = 'down' if dy>0 else 'up'
            else:
                action_name = np.random.choice(actions)
            action_idx = actions.index(action_name)
            # --- Thực hiện hành động ---
            next_state, reward, done, _ = env.step(action_idx)

            # --- Lưu vào episode_buffer ---
            env.episode_buffer.append((full_state, action_name, reward))

            # --- Cập nhật Q-table theo First-Visit MC (duyệt xuôi) ---
            visited_pairs = set()
            for t, (s, a, r) in enumerate(env.episode_buffer):
                G = sum((gamma**(k-t)) * env.episode_buffer[k][2] for k in range(t, len(env.episode_buffer)))
                if (s, a) not in visited_pairs:
                    mc_Q[s][a] += alpha * (G - mc_Q[s][a])
                    visited_pairs.add((s, a))

            # --- Reset buffer nếu done ---
            if done or env.steps >= env.max_steps:
                env.episode_buffer = []

            # --- Cập nhật epsilon ---
            epsilon = max(0.1, epsilon * 0.995)
            # --- Cập nhật state ---
            state_xy = next_state
            done = done or env.steps >= env.max_steps or env.state == env.goal


        
        elif algo == "Q-learning":
            if np.random.rand() < epsilon:
                action_name = np.random.choice(actions)
            else:
                action_name = max(ql_Q[full_state], key=ql_Q[full_state].get)
            
            action_idx = actions.index(action_name)
            
            next_state, r, done, _ = env.step(action_idx)
            
            next_visited_code = encode_visited(env.waypoints, env.visited_waypoints)
            next_state_tuple = (next_state[0], next_state[1], next_visited_code)
            
            ql_Q[full_state][action_name] += alpha * (
                r + gamma * max(ql_Q[next_state_tuple].values()) - ql_Q[full_state][action_name]
            )
            
            state_xy = next_state
            reward = r
            epsilon = max(0.1, epsilon * 0.995)
            
        elif algo == "Q-learning":
            # 1. Chọn hành động bằng Abstract State hiện tại (epsilon-greedy)
            if np.random.rand() < epsilon:
                action_name = np.random.choice(actions)
            else:
                # Dùng abstract_state làm key để chọn hành động tối ưu
                action_name = max(ql_Q[abstract_state], key=ql_Q[abstract_state].get) 
            
            action_idx = actions.index(action_name)
            
            # 2. Thực hiện hành động
            next_state, r, done, _ = env.step(action_idx)
            
            # 3. Lấy next_abstract_state sau khi bước
            next_abstract_state = get_abstract_state(env) 
            
            # 4. Cập nhật Q-value (dùng next_abstract_state để tìm max Q)
            # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
            ql_Q[abstract_state][action_name] += alpha * (
                r + gamma * max(ql_Q[next_abstract_state].values()) - ql_Q[abstract_state][action_name]
            )
            
            state_xy = next_state
            reward = r
            
            # 5. Giảm Epsilon (Sử dụng decay rate mới)
            epsilon = max(0.01, epsilon * epsilon_decay_rate)

        elif algo == "A2C":
            # Chuyển state (grid) sang tensor để đưa vào model
            state_tensor = env.build_grid_state().unsqueeze(0)
            a2c_model.eval()
            with torch.no_grad():
                # Lấy logits từ policy head
                policy_logits, _ = a2c_model(state_tensor)
                # Tính xác suất và lấy mẫu action (Exploitation)
                action_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
                action_idx = torch.multinomial(action_probs, 1).item()
            
            next_state, r, done, _ = env.step(action_idx)
            state_xy = next_state

            _save_and_reload_a2c(a2c_model, a2c_model_file)

        else:
            return {"error": f"Unknown algorithm: {algo}"}

        return {
            "algorithm": algo,
            "state": state_xy,
            "reward": reward,
            "final_reward": reward,
            "done": done,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints)
        }

# ---------------------------
# Run A* Algorithm (unchanged)
# ---------------------------

@app.post("/run_a_star")
def run_a_star(req: AStarRequest):
    """Tìm đường đi tối ưu và thực hiện theo đường đi đó."""
    with _env_lock:
        start_time = time.time()
        rewards_over_time = []
        
        start = env.get_state()
        
        path = plan_path_through_waypoints(start, env.waypoints, req.goal or env.goal,
                                           env.obstacles, env.width, env.height)
        if not path:
            return {"error": "Không tìm thấy đường đi qua tất cả waypoint"}
        
        env.reset()
        total_reward = 0
        for node in path[1:]:
            s, r, done, info = env.step_to(node)
            total_reward += r
            last_r = r
            rewards_over_time.append(total_reward)

        done = (env.state == env.goal and len(env.visited_waypoints) == len(env.waypoints))
        elapsed_time = time.time() - start_time

        return {
            "algorithm": "A*", "path": path, "state": env.get_state(),
            "reward": total_reward, "done": done, "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints), "info": {},
            "ascii": env.render_ascii(), "elapsed_time": elapsed_time,
            "rewards_over_time": rewards_over_time
        }

# ---------------------------
# Save Endpoints
# ---------------------------
@app.post("/save_qlearning")
def save_qlearning():
    with open(os.path.join(models_dir, 'qlearning_qtable.pkl'), 'wb') as f:
        pickle.dump(ql_Q, f)
    return {"status": "Q-learning Q-table saved"}

@app.post("/reset_qlearning")
def reset_qlearning():
    global ql_Q
    ql_Q = defaultdict(_new_qrow)
    # Delete old Q-table file
    qlearning_file = os.path.join(models_dir, 'qlearning_qtable.pkl')
    if os.path.exists(qlearning_file):
        os.remove(qlearning_file)
    return {"status": "Q-learning Q-table reset - ready for fresh training"}

@app.post("/save_mc")
def save_mc():
    with open(os.path.join(models_dir, 'mc_qtable.pkl'), 'wb') as f:
        pickle.dump(mc_Q, f)
    return {"status": "MC Q-table saved"}


@app.post("/save_sarsa")
def save_sarsa():
    with open(os.path.join(models_dir, 'sarsa_qtable.pkl'), 'wb') as f:
        pickle.dump(sarsa_Q, f)
    return {"status": "SARSA Q-table saved"}

@app.post("/save_a2c")
def save_a2c():
    torch.save(a2c_model.state_dict(), os.path.join(models_dir, 'a2c_model.pth'))
    return {"status": "A2C model saved"}

@app.get("/qlearning_stats")
def qlearning_stats():
    """Get Q-Learning statistics for debugging"""
    q_size = len(ql_Q)
    sample_entries = []
    
    if q_size > 0:
        # Get some sample Q-values to inspect learning
        sample_states = list(ql_Q.keys())[:5]
        for state in sample_states:
            q_values = ql_Q[state]
            best_action = max(q_values, key=q_values.get)
            sample_entries.append({
                "state": state,
                "best_action": best_action,
                "q_values": dict(q_values)
            })
    
    return {
        "q_table_size": q_size,
        "current_epsilon": epsilon,
        "sample_entries": sample_entries,
        "learning_params": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay
        }
    }

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/index.html")

@app.post("/train_qlearning_fast")
def train_qlearning_fast():
    """Fast Q-Learning training session (100 episodes)"""
    global ql_Q
    
    # Reset Q-table for fresh training
    ql_Q = defaultdict(_new_qrow)
    
    # Fast training parameters
    train_episodes = 100
    train_epsilon = 0.5
    train_alpha = 0.2
    
    successful_episodes = 0
    
    for episode in range(train_episodes):
        # Reset environment
        temp_env = GridWorldEnv(env.width, env.height, env.start, env.goal, 
                               list(env.obstacles), env.waypoints, max_steps=200)
        for k, v in REWARD_PARAMS.items():
            setattr(temp_env, k, v)
        
        state = tuple(temp_env.reset())
        done = False
        episode_reward = 0
        
        while not done and temp_env.steps < 200:
            # Choose action with epsilon-greedy + heuristic
            if state not in ql_Q:
                ql_Q[state] = _new_qrow()
            
            if np.random.rand() < train_epsilon:
                # Heuristic-guided exploration
                x, y = state
                goal_x, goal_y = temp_env.goal
                best_action = None
                best_distance = float('inf')
                
                for action in actions:
                    action_idx = actions.index(action)
                    dx, dy = temp_env.ACTIONS[action_idx]
                    new_x, new_y = x + dx, y + dy
                    
                    if (0 <= new_x < temp_env.width and 0 <= new_y < temp_env.height and 
                        (new_x, new_y) not in temp_env.obstacles):
                        distance = abs(new_x - goal_x) + abs(new_y - goal_y)
                        if distance < best_distance:
                            best_distance = distance
                            best_action = action
                
                action = best_action or np.random.choice(actions)
            else:
                # Exploitation
                action = max(ql_Q[state], key=ql_Q[state].get)
            
            # Take action
            action_idx = actions.index(action)
            next_state, reward, done_raw, _ = temp_env.step(action_idx)
            
            # Enhanced reward shaping
            old_state = state
            new_state = tuple(next_state)
            reward =  reward
            
            # Goal guard
            reached_goal_now = (new_state == temp_env.goal)
            if reached_goal_now and len(temp_env.visited_waypoints) < len(temp_env.waypoints):
                reward += REWARD_PARAMS["goal_before_waypoints_penalty"]
            
            done = (new_state == temp_env.goal and len(temp_env.visited_waypoints) == len(temp_env.waypoints))
            episode_reward += reward
            
            # Q-Learning update
            if new_state not in ql_Q:
                ql_Q[new_state] = _new_qrow()
            
            ql_Q[state][action] += train_alpha * (
                reward + gamma * max(ql_Q[new_state].values()) - ql_Q[state][action]
            )
            
            state = new_state
        
        if done:
            successful_episodes += 1
        
        # Decay epsilon
        train_epsilon = max(0.1, train_epsilon * 0.995)
    
    # Save trained Q-table
    with open(os.path.join(models_dir, 'qlearning_qtable.pkl'), 'wb') as f:
        pickle.dump(ql_Q, f)
    
    return {
        "status": "Fast Q-Learning training completed",
        "episodes_trained": train_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": successful_episodes / train_episodes,
        "q_table_size": len(ql_Q),
        "message": f"Trained {train_episodes} episodes with {successful_episodes} successes ({successful_episodes/train_episodes:.1%} success rate)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
