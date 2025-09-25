# app/server.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
from threading import Lock
import os, pickle, torch, numpy as np, time, heapq
from itertools import permutations
from collections import defaultdict
import torch.nn.functional as F

from app.robot_env import GridWorldEnv
from clients.train_a2c import ActorCritic

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="RL Robot API", version="1.1.1")
_env_lock = Lock()
app.mount("/web", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../clients/web")), name="web")

# ---------------------------
# Environment + Reward params (thống nhất)
# ---------------------------
GRID_W, GRID_H = 10, 8
START = (0, 0)
GOAL = (9, 7)
WAYPOINTS = [(3, 2), (6, 5)]
OBSTACLES = [(1, 1), (2, 3), (4, 4), (5, 1), (7, 6)]

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
# Q-tables (LOAD SỚM, TOÀN CỤC)
# ---------------------------
actions = ['up', 'right', 'down', 'left']

def _new_qrow():
    return {a: 0.0 for a in actions}

def _load_qtable(path: str):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            d = defaultdict(_new_qrow)
            d.update(loaded)
            return d
        except Exception as e:
            print(f"⚠️ Không load được Q-table {path}: {e}. Dùng defaultdict rỗng.")
    return defaultdict(_new_qrow)

mc_qfile     = os.path.join(models_dir, "mc_qtable.pkl")
ql_qfile     = os.path.join(models_dir, "qlearning_qtable.pkl")
sarsa_qfile  = os.path.join(models_dir, "sarsa_qtable.pkl")

mc_Q    = _load_qtable(mc_qfile)
ql_Q    = _load_qtable(ql_qfile)
sarsa_Q = _load_qtable(sarsa_qfile)

# ---------------------------
# Load A2C model
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
        a2c_model.load_state_dict(torch.load(a2c_model_file))
        a2c_model.eval()
        print("✅ A2C model loaded successfully")
    except Exception as e:
        print(f"⚠️ Không load được A2C checkpoint: {e}. Dùng model mới.")

# ---------------------------
# Simple Q-Learning parameters
# ---------------------------
alpha, gamma = 0.1, 0.9     # Standard learning parameters
epsilon = 1.0                # Start with full exploration  
epsilon_min = 0.01          # Very low minimum for strong exploitation
epsilon_decay = 0.995       # Gradual decay

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

# ---------------------------
# A* helpers
# ---------------------------
def a_star(start, goal, obstacles, width, height):
    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = []
    heapq.heappush(open_set, (h(start, goal), 0, start, [start]))
    visited = set()
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
                heapq.heappush(open_set, (g + 1 + h((nx, ny), goal), g + 1, (nx, ny), path + [(nx, ny)]))
    return []

def plan_path_through_waypoints(start, waypoints, goal, obstacles, width, height):
    best_path, min_len = None, float('inf')
    for order in permutations(waypoints):
        path, curr, valid = [], start, True
        for wp in order:
            sub = a_star(curr, wp, obstacles, width, height)
            if not sub: valid = False; break
            path += sub[:-1]; curr = wp
        if not valid: continue
        sub = a_star(curr, goal, obstacles, width, height)
        if not sub: continue
        path += sub
        if len(path) < min_len:
            min_len, best_path = len(path), path
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
# Endpoints
# ---------------------------
@app.get("/map")
def get_map():
    with _env_lock:
        return {"map": env.get_map()}

@app.post("/reset")
def reset(req: ResetRequest):
    global env, epsilon
    with _env_lock:
        w = req.width or env.width
        h = req.height or env.height
        s = req.start or env.start
        g = req.goal or env.goal
        wp = req.waypoints if req.waypoints is not None else list(env.waypoints)
        ob = req.obstacles if req.obstacles is not None else list(env.obstacles)
        ms = req.max_steps if req.max_steps is not None else DEFAULT_MAX_STEPS

        env = GridWorldEnv(w, h, s, g, ob, wp, max_steps=ms)
        for k, v in REWARD_PARAMS.items():
            setattr(env, k, v)

        # Reset epsilon for new learning session
        epsilon = 0.3  # Start with balanced exploration
        state = env.reset(max_steps=ms)
        return {"state": state, "map": env.get_map(), "ascii": env.render_ascii()}

import random

# ---------------------------
# Reset All API
# ---------------------------
@app.post("/reset_all")
def reset_all():
    global env
    with _env_lock:
        # Random lại obstacles, waypoints và goal
        w, h = env.width, env.height
        start = (0, 0)

        # Random obstacles
        all_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) != start]
        random.shuffle(all_cells)
        obstacles = all_cells[:8]   # ví dụ chọn 8 chướng ngại vật

        # Random 2 waypoint + 1 goal
        remain = [cell for cell in all_cells if cell not in obstacles]
        waypoints = remain[:2]
        goal = remain[2]

        # Tạo môi trường mới
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
    with _env_lock:
        if inp.action_name is not None:
            s, r, done_raw, info = env.step_by_name(inp.action_name)
        elif inp.action is not None:
            s, r, done_raw, info = env.step(inp.action)
        else:
            return {"error": "No action provided"}

        reached_goal_now = (tuple(s) == env.goal)
        r = _apply_goal_guard_and_unified_reward(r, reached_goal_now)
        done = (tuple(s) == env.goal and len(env.visited_waypoints) == len(env.waypoints)) or env.steps >= env.max_steps

        return {
            "state": s,
            "reward": r,
            "final_reward": r,
            "done": done,
            "info": info,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "travel_time_ms": env.steps * 500,
            "ascii": env.render_ascii()
        }

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
    global epsilon, mc_Q, ql_Q, sarsa_Q
    algo = req.algorithm
    with _env_lock:
        state_xy = env.get_state()
        visited_code = _encode_visited(env.waypoints, env.visited_waypoints)
        full_state = (state_xy[0], state_xy[1], visited_code)

        reward = 0.0
        done = False

        if algo == "MC":
            action_name = _select_action_from(mc_Q, full_state, epsilon)
            action_idx = actions.index(action_name)
            next_state, r, done_raw, _ = env.step(action_idx)

            reached_goal_now = (tuple(next_state) == env.goal)
            r = _apply_goal_guard_and_unified_reward(r, reached_goal_now)
            done = (tuple(next_state) == env.goal and len(env.visited_waypoints) == len(env.waypoints)) or env.steps >= env.max_steps

            next_code = _encode_visited(env.waypoints, env.visited_waypoints)
            next_full_state = (next_state[0], next_state[1], next_code)

            G = r + gamma * max(mc_Q[next_full_state].values())
            mc_Q[full_state][action_name] += alpha * (G - mc_Q[full_state][action_name])

            mc_Q = _save_and_reload_q(mc_Q, mc_qfile)

            reward = r
            state_xy = next_state
            # Optimized epsilon decay for Monte Carlo
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        elif algo == "Q-learning":
            # Sử dụng simple state (x, y) giống như trong train_qlearning.py
            simple_state = tuple(state_xy)
            action_name = _select_action_from(ql_Q, simple_state, epsilon)
            action_idx = actions.index(action_name)
            
            old_state = tuple(state_xy)
            next_state, r, done_raw, _ = env.step(action_idx)
            new_state = tuple(next_state)

            # Apply simple reward shaping for Q-Learning
            r = _apply_simple_reward_shaping(old_state, new_state, r)
            
            reached_goal_now = (new_state == env.goal)
            r = _apply_goal_guard_and_unified_reward(r, reached_goal_now)
            done = (new_state == env.goal and len(env.visited_waypoints) == len(env.waypoints)) or env.steps >= env.max_steps

            next_simple_state = new_state

            ql_Q[simple_state][action_name] += alpha * (
                r + gamma * max(ql_Q[next_simple_state].values()) - ql_Q[simple_state][action_name]
            )

            ql_Q = _save_and_reload_q(ql_Q, ql_qfile)

            reward = r
            state_xy = next_state
            # Optimized epsilon decay for Q-learning
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        elif algo == "SARSA":
            action_name = _select_action_from(sarsa_Q, full_state, epsilon)
            action_idx = actions.index(action_name)
            next_state, r, done_raw, _ = env.step(action_idx)

            reached_goal_now = (tuple(next_state) == env.goal)
            r = _apply_goal_guard_and_unified_reward(r, reached_goal_now)
            done = (tuple(next_state) == env.goal and len(env.visited_waypoints) == len(env.waypoints)) or env.steps >= env.max_steps

            next_code = _encode_visited(env.waypoints, env.visited_waypoints)
            next_full_state = (next_state[0], next_state[1], next_code)

            next_action_name = _select_action_from(sarsa_Q, next_full_state, epsilon)

            sarsa_Q[full_state][action_name] += alpha * (
                r + gamma * sarsa_Q[next_full_state][next_action_name] - sarsa_Q[full_state][action_name]
            )

            sarsa_Q = _save_and_reload_q(sarsa_Q, sarsa_qfile)

            reward = r
            state_xy = next_state
            # Optimized epsilon decay for SARSA
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            epsilon = max(0.1, epsilon * 0.995)
            
        elif algo == "SARSA":
            # Chọn hành động A từ trạng thái S theo chính sách epsilon-greedy
            if np.random.rand() < epsilon:
                action_name = np.random.choice(actions)
            else:
                action_name = max(sarsa_Q[full_state], key=sarsa_Q[full_state].get)
            
            action_idx = actions.index(action_name)
            
            # Thực hiện hành động A, nhận S' và R
            next_state, r, done, _ = env.step(action_idx)
            
            next_visited_code = encode_visited(env.waypoints, env.visited_waypoints)
            next_state_tuple = (next_state[0], next_state[1], next_visited_code)

            # Chọn hành động tiếp theo A' từ S' theo chính sách epsilon-greedy
            if np.random.rand() < epsilon:
                next_action_name = np.random.choice(actions)
            else:
                next_action_name = max(sarsa_Q[next_state_tuple], key=sarsa_Q[next_state_tuple].get)
            
            # Cập nhật Q-table theo công thức SARSA
            sarsa_Q[full_state][action_name] += alpha * (
                r + gamma * sarsa_Q[next_state_tuple][next_action_name] - sarsa_Q[full_state][action_name]
            )

            state_xy = next_state
            reward = r
            epsilon = max(0.1, epsilon * 0.995)

        elif algo == "A2C":
            state_tensor = env.build_grid_state().unsqueeze(0)
            a2c_model.eval()
            with torch.no_grad():
                policy_logits, _ = a2c_model(state_tensor)
                action_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
                action_idx = torch.multinomial(action_probs, 1).item()

            next_state, r, done_raw, _ = env.step(action_idx)

            reached_goal_now = (tuple(next_state) == env.goal)
            r = _apply_goal_guard_and_unified_reward(r, reached_goal_now)
            done = (tuple(next_state) == env.goal and len(env.visited_waypoints) == len(env.waypoints)) or env.steps >= env.max_steps

            reward = r
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
            "visited_waypoints": list(env.visited_waypoints),
            "travel_time_ms": env.steps * 500,
            "max_steps": env.max_steps
        }

# ---------------------------
# A* step với reward RL thống nhất
# ---------------------------
def step_to_rl(self, target):
    self.state = target
    self.steps += 1
    reward = self.step_penalty

    if target in self.obstacles:
        reward += self.obstacle_penalty

    if target in self.waypoints and target not in self.visited_waypoints:
        self.visited_waypoints.add(target)
        reward += self.waypoint_reward

    reached_goal_now = (target == self.goal)
    if reached_goal_now and len(self.visited_waypoints) == len(self.waypoints):
        reward += self.goal_reward
        done = True
    else:
        if reached_goal_now and len(self.visited_waypoints) < len(self.waypoints):
            reward += self.goal_before_waypoints_penalty
        done = False

    info = {"note": "Auto move by A* (RL reward unified)"}
    return target, reward, done, info

GridWorldEnv.step_to = step_to_rl

# ---------------------------
# A* endpoint
# ---------------------------
@app.post("/run_a_star")
def run_a_star(req: AStarRequest):
    with _env_lock:
        start_time = time.time()
        rewards_over_time = []

        start_pos = env.get_state()
        path = plan_path_through_waypoints(
            start_pos, env.waypoints, req.goal or env.goal, env.obstacles, env.width, env.height
        )
        if not path:
            return {"error": "Không tìm thấy đường đi qua tất cả waypoint"}

        env.reset()
        for k, v in REWARD_PARAMS.items():
            setattr(env, k, v)

        total_reward = 0.0
        last_r = 0.0
        for node in path[1:]:
            s, r, done, info = env.step_to(node)
            total_reward += r
            last_r = r
            rewards_over_time.append(total_reward)

        done = (env.state == env.goal and len(env.visited_waypoints) == len(env.waypoints))
        elapsed_time = time.time() - start_time

        return {
            "algorithm": "A*",
            "path": path,
            "state": env.get_state(),
            "reward": total_reward,
            "final_reward": last_r,
            "done": done,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "travel_time_ms": env.steps * 500,
            "ascii": env.render_ascii(),
            "elapsed_time": elapsed_time,
            "rewards_over_time": rewards_over_time
        }

# ---------------------------
# Save endpoints
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
            reward = _apply_enhanced_reward_shaping(old_state, new_state, reward)
            
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
