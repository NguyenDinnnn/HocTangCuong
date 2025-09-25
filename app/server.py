from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
from threading import Lock
import os, pickle, torch, numpy as np, time
import heapq
from itertools import permutations
from collections import defaultdict
import torch.nn.functional as F

from app.robot_env import GridWorldEnv
from clients.train_a2c import ActorCritic

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="RL Robot API", version="1.0.0")
_env_lock = Lock()
app.mount("/web", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../clients/web")), name="web")

# ---------------------------
# Environment
# ---------------------------
width, height = 10, 8
start = (0,0)
goal = (9,7)
waypoints = [(3,2),(6,5)]
obstacles = [(1,1),(2,3),(4,4),(5,1),(7,6)]
# CẬP NHẬT: Đặt giá trị mặc định cho max_steps
env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=500)

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
height, width = env.height, env.width
n_actions = len(env.ACTIONS)
a2c_model = ActorCritic(in_channels, height, width, n_actions)
if os.path.exists(a2c_model_file):
    try:
        a2c_model.load_state_dict(torch.load(a2c_model_file))
        a2c_model.eval()
        print("✅ A2C model loaded successfully")
    except RuntimeError:
        print("⚠️ Không load được A2C checkpoint. Sẽ dùng model mới.")

# ---------------------------
# RL params
# ---------------------------
actions = ['up', 'right', 'down', 'left']
alpha, gamma = 0.1, 0.99
epsilon = 1.0

# ---------------------------
# Request Models
# ---------------------------
class ResetRequest(BaseModel):
    width: Optional[int]=None
    height: Optional[int]=None
    start: Optional[Tuple[int,int]]=None
    goal: Optional[Tuple[int,int]]=None
    waypoints: Optional[List[Tuple[int,int]]]=None
    obstacles: Optional[List[Tuple[int,int]]]=None
    max_steps: Optional[int]=None

class ActionInput(BaseModel):
    action: Optional[int]=None
    action_name: Optional[str]=None

class AlgorithmRequest(BaseModel):
    algorithm: str

class AStarRequest(BaseModel):
    goal: Optional[Tuple[int,int]] = None

# ---------------------------
# A* functions
# ---------------------------
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
# Extend GridWorldEnv for A* step (RL-style reward)
# ---------------------------
def step_to_rl(self, target):
    self.state = target
    self.steps += 1
    reward = -0.1
    done = False
    if target in self.waypoints and target not in self.visited_waypoints:
        self.visited_waypoints.add(target)
        reward = 1
    if target == self.goal and len(self.visited_waypoints) == len(self.waypoints):
        done = True
        reward = 10
    info = {"note": "Auto move by A* (RL reward)"}
    return target, reward, done, info

GridWorldEnv.step_to = step_to_rl

# ---------------------------
# API Endpoints
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
        state = env.reset(max_steps=ms)
        return {"state": state, "map": env.get_map(), "ascii": env.render_ascii()}

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
# Run RL Algorithm step by step
# ---------------------------
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
        
        visited_code = encode_visited(env.waypoints, env.visited_waypoints)
        full_state = (state_xy[0], state_xy[1], visited_code)

        if algo == "MC":
            if np.random.rand() > epsilon:
                if full_state in mc_Q and any(mc_Q[full_state].values()):
                     action_name = max(mc_Q[full_state], key=mc_Q[full_state].get)
                else:
                     action_name = np.random.choice(actions)
            else:
                action_name = np.random.choice(actions)
            action_idx = actions.index(action_name)
            
            next_state, r, done, _ = env.step(action_idx)
            
            # Update for MC after each step
            # Đây là sự thay đổi lớn trong cách MC học, từ offline sang online, nhưng nó giúp mô phỏng từng bước
            G = r + gamma * max(mc_Q[next_state].values())
            mc_Q[full_state][action_name] += alpha * (G - mc_Q[full_state][action_name])
            
            reward = r
            state_xy = next_state
        
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
            
            next_state, r, done, _ = env.step(action_idx)
            state_xy = next_state
            reward = r

        return {
            "state": state_xy,
            "reward": reward,
            "done": done or env.steps >= env.max_steps,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints)
        }

# ---------------------------
# Run A* Algorithm (unchanged)
# ---------------------------
@app.post("/run_a_star")
def run_a_star(req: AStarRequest):
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
            rewards_over_time.append(total_reward)

        done = (env.state == env.goal and len(env.visited_waypoints) == len(env.waypoints))
        elapsed_time = time.time() - start_time
        
        return {
            "algorithm": "A*",
            "path": path,
            "state": env.get_state(),
            "reward": total_reward,
            "done": done,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "info": {},
            "ascii": env.render_ascii(),
            "elapsed_time": elapsed_time,
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

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
