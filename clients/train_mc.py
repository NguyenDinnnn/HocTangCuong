import sys, os, pickle
import numpy as np
from collections import defaultdict
from datetime import datetime
import random 
import math 
from typing import Tuple, List, Set, Optional

# Giả định đường dẫn import đã đúng
try:
    # Cần đảm bảo cấu trúc thư mục đúng: clients/models/train_qlearning...py -> ../app/robot_env.py
    # Thêm thư mục gốc vào path để import GridWorldEnv
    # Sử dụng os.path.join để đảm bảo tính tương thích với mọi hệ điều hành
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from app.robot_env import GridWorldEnv
except ImportError:
    print("WARNING: Không thể import GridWorldEnv. Vui lòng kiểm tra đường dẫn.")
    
    # --------------------------------------------------------------------------
    # Lớp giả lập GridWorldEnv để trả về đúng kiểu dữ liệu
    # --------------------------------------------------------------------------
    class GridWorldEnv: 
        ACTION_NAMES = ["up", "right", "down", "left"]
        ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        def __init__(self, **kwargs): pass
        def reset(self, **kwargs): 
            self.state = (0, 0)
            self.steps = 0
            self.visited_waypoints = set()
            self.max_steps = kwargs.get('max_steps', 500)
            return self.state
        def get_state(self): return self.state
        def is_done(self): return self.steps >= self.max_steps
        def step(self, action_idx): 
            self.steps += 1
            # Giả lập reward và done
            return (0, 0), -0.5, self.steps >= self.max_steps, {}
        
        # Sửa lại __getattr__ để trả về các giá trị/kiểu dữ liệu mặc định an toàn
        def __getattr__(self, name):
            if name in ['width', 'height', 'steps', 'max_steps']: return 10 if name in ['width', 'height'] else 0
            if name in ['state', 'goal']: return (0,0) 
            if name in ['waypoints', 'visited_waypoints', 'obstacles']: 
                return set() if name == 'visited_waypoints' or name == 'obstacles' else []
            if name.endswith('penalty') or name.endswith('reward'): return 0 
            return super().__getattr__(name)
    
    # Gán các thuộc tính cơ bản cho lớp giả lập 
    GridWorldEnv.width = 10
    GridWorldEnv.height = 8
    GridWorldEnv.state = (0, 0)
    GridWorldEnv.goal = (9, 7)
    GridWorldEnv.waypoints = [(3, 2), (6, 5)]
    GridWorldEnv.visited_waypoints = set()
    GridWorldEnv.obstacles = set()
    GridWorldEnv.steps = 0
    GridWorldEnv.max_steps = 500
    GridWorldEnv.step_penalty = -0.5
    
    # --------------------------------------------------------------------------


# ==============================================================================
# STATE ABSTRACTION FUNCTIONS (GIỮ NGUYÊN)
# ==============================================================================

def encode_visited(wp_list: List[Tuple[int, int]], visited_set: Set[Tuple[int, int]]) -> int:
    """Mã hóa trạng thái Waypoint đã thăm dưới dạng một số nguyên (bitmask)."""
    code = 0
    for i, wp in enumerate(wp_list): 
        if wp in visited_set:
            code |= (1 << i)
    return code

def get_abstract_state(env: GridWorldEnv) -> Tuple:
    """
    Tạo trạng thái trừu tượng (State Abstraction) từ môi trường.
    Trạng thái trừu tượng: (Goal_Sign_X, Goal_Sign_Y, Waypoint_Sign_X, Waypoint_Sign_Y, Obstacle_Code, Visited_Code)
    """
    rx, ry = env.state
    
    # 1. Tín hiệu Goal (Relative Goal Sign)
    gx, gy = env.goal
    dx_goal = gx - rx
    dy_goal = gy - ry
    sign_x_goal = int(math.copysign(1, dx_goal)) if dx_goal != 0 else 0
    sign_y_goal = int(math.copysign(1, dy_goal)) if dy_goal != 0 else 0

    # 2. Tín hiệu Waypoint gần nhất chưa thăm (Relative Waypoint Sign)
    closest_wp = None
    min_dist = float('inf')
    
    for wp in env.waypoints:
        if wp not in env.visited_waypoints:
            dist = abs(wp[0] - rx) + abs(wp[1] - ry)
            if dist < min_dist:
                min_dist = dist
                closest_wp = wp
    
    sign_x_wp, sign_y_wp = 0, 0
    if closest_wp:
        dx_wp = closest_wp[0] - rx
        dy_wp = closest_wp[1] - ry
        sign_x_wp = int(math.copysign(1, dx_wp)) if dx_wp != 0 else 0
        sign_y_wp = int(math.copysign(1, dy_wp)) if dy_wp != 0 else 0

    # 3. Mã chướng ngại vật cục bộ (Local Obstacle Code)
    obstacle_code = 0
    for i, (dx, dy) in enumerate(GridWorldEnv.ACTIONS):
        next_pos = (rx + dx, ry + dy)
        # Kiểm tra tường hoặc chướng ngại vật
        if next_pos in env.obstacles or not (0 <= next_pos[0] < env.width and 0 <= next_pos[1] < env.height):
            obstacle_code |= (1 << i)
            
    # 4. Mã Waypoint đã thăm (Visited Code)
    visited_code = encode_visited(env.waypoints, env.visited_waypoints)

    return (sign_x_goal, sign_y_goal, sign_x_wp, sign_y_wp, obstacle_code, visited_code)

# ==============================================================================
# HÀM NGẪU NHIÊN HÓA MÔI TRƯỜNG (CHO MÔI TRƯỜNG ĐỘNG)
# ==============================================================================

def randomize_env_params(width: int, height: int, num_waypoints: int = 2, min_obs: int = 5, max_obs: int = 10) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Tạo ngẫu nhiên vị trí bắt đầu, đích, waypoint và chướng ngại vật không trùng lặp."""
    all_coords = set((x, y) for x in range(width) for y in range(height))
    
    required_num = 2 + num_waypoints 
    
    if len(all_coords) < required_num:
        raise ValueError("Lưới quá nhỏ để chọn đủ vị trí bắt đầu, đích, và waypoint.")
        
    required_positions = random.sample(list(all_coords), required_num)
    
    start = required_positions[0]
    goal = required_positions[1]
    waypoints = required_positions[2:]
    
    available_coords = all_coords - set(required_positions)
    
    num_obstacles = random.randint(min_obs, min(max_obs, len(available_coords)))
    obstacles = set(random.sample(list(available_coords), num_obstacles))

    return start, goal, waypoints, obstacles


# ==============================================================================
# 1. THAM SỐ RL VÀ KHỞI TẠO (ĐÃ TỐI ƯU HÓA)
# ==============================================================================

EPISODES = 100000
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01  
ALPHA = 0.1 

# TỐI ƯU: Tăng Gamma để agent siêu xa thị (Super long-term vision)
GAMMA = 0.995

# TỐI ƯU: Giảm Epsilon decay rate để tăng thời gian khám phá
EPSILON_DECAY_RATE = 0.998 

# Tham số môi trường
WIDTH, HEIGHT = 10, 8
MAX_STEPS = 500

# REWARD TỐI ƯU cho ĐƯỜNG ĐI NGẮN NHẤT & THỨ TỰ WAYPOINT
REWARDS = {
    # TỐI ƯU: Tăng phạt bước đi để buộc tìm đường ngắn (Shortest path enforcement)
    "step_penalty": -1.0, 
    "wall_penalty": -20,
    "obstacle_penalty": -50, 
    # TỐI ƯU: Tăng thưởng Waypoint để cân bằng với Goal Reward (Waypoint Priority)
    "waypoint_reward": 100, 
    "goal_reward": 200,
    "goal_before_waypoints_penalty": -50, 
}

# Khởi tạo Môi trường với các tham số CỐ ĐỊNH BAN ĐẦU
START, GOAL = (0,0), (9,7)
WAYPOINTS = [(3,2),(6,5)]
OBSTACLES = [(1,1),(2,3),(4,4),(5,1),(7,6)]

env = GridWorldEnv(width=WIDTH, height=HEIGHT, start=START, goal=GOAL, 
                   obstacles=OBSTACLES, waypoints=WAYPOINTS, max_steps=MAX_STEPS)

# Áp dụng các tham số Reward đã tối ưu vào môi trường
for k, v in REWARDS.items():
    if hasattr(env, k):
        setattr(env, k, v)
        
actions = env.ACTION_NAMES
# Q-table TD (Q-learning) sử dụng State Abstraction
Q = defaultdict(lambda: {a: 0.0 for a in actions}) 

# Đường dẫn lưu model
model_dir = os.path.join(os.path.dirname(__file__), "clients/models") 
os.makedirs(model_dir, exist_ok=True)
q_file = os.path.join(model_dir, "mc_qtable.pkl")

# Load model (nếu có)
start_episode = 0
epsilon = INITIAL_EPSILON
if os.path.exists(q_file):
    try:
        with open(q_file, "rb") as f:
            data = pickle.load(f)
        
        Q.update(data['Q'])
        epsilon = data.get('epsilon', INITIAL_EPSILON)
        start_episode = data.get('episode', 0) + 1
        print(f"✅ Đã tải mô hình Q-learning (Abstraction) từ episode {start_episode - 1}. Epsilon: {epsilon:.6f}")
    except Exception as e:
        print(f"⚠️ Không thể tải mô hình: {e}. Bắt đầu train lại từ đầu.")

# ==============================================================================
# 2. HÀM CHỌN HÀNH ĐỘNG VÀ HUẤN LUYỆN
# ==============================================================================
def choose_action(state_tuple, current_epsilon):
    """Lựa chọn hành động theo chính sách Epsilon-Greedy."""
    if np.random.rand() < current_epsilon:
        # Exploration: Chọn ngẫu nhiên
        return np.random.choice(actions) 
    else:
        # Exploitation: Chọn hành động có Q-value lớn nhất
        q_values = Q[state_tuple]
        if all(q == 0.0 for q in q_values.values()):
             # Nếu tất cả đều bằng 0, chọn ngẫu nhiên
             return np.random.choice(actions)
            
        max_q = max(q_values.values())
        # Chọn ngẫu nhiên trong số các hành động tốt nhất (break ties)
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)

def train_qlearning_with_abstraction(): 
    global epsilon
    start_time = datetime.now()
    
    success_count = 0
    
    for episode in range(start_episode, EPISODES):
        
        # --- BƯỚC 1: NGẪU NHIÊN HÓA MÔI TRƯỜNG ---
        try:
            start, goal, waypoints, obstacles = randomize_env_params(
                WIDTH, HEIGHT, num_waypoints=2, min_obs=5, max_obs=10
            )
        except ValueError as e:
            continue 
        
        # Reset môi trường với tham số ngẫu nhiên
        env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints) 
        
        # Lấy trạng thái trừu tượng ban đầu
        abstract_state = get_abstract_state(env) 
        done = False
        total_reward = 0
        
        # --- BẮT ĐẦU EPISODE ---
        while not done:
            # 1. Chọn Hành động
            action_name = choose_action(abstract_state, epsilon)
            
            old_abstract_state = abstract_state
            
            # 2. Thực hiện Hành động
            action_idx = env.ACTION_NAMES.index(action_name) 
            _, reward, _, _ = env.step(action_idx)
            
            # 3. Lấy Trạng thái MỚI
            next_abstract_state = get_abstract_state(env)
            
            # Kích hoạt done khi đạt mục tiêu hoặc hết max_steps
            done = env.is_done() or env.steps >= env.max_steps
            
            # 4. CẬP NHẬT Q-VALUE (Q-learning)
            max_q_next = 0.0
            if not done:
                max_q_next = max(Q[next_abstract_state].values())
            
            # G: Return ước tính (r + gamma * max(Q(s',a')))
            G = reward + GAMMA * max_q_next
            
            # Cập nhật Q-value: Q(s,a) += alpha * (G - Q(s,a))
            Q[old_abstract_state][action_name] += ALPHA * (G - Q[old_abstract_state][action_name])
            
            # Chuyển trạng thái
            abstract_state = next_abstract_state
            total_reward += reward
        
        # --- KẾT THÚC EPISODE ---
        
        is_success = env.state == env.goal and set(env.waypoints).issubset(env.visited_waypoints)
        if is_success:
            success_count += 1
            
        # Giảm Epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY_RATE)
        
        # In kết quả và lưu model
        if (episode + 1) % 1000 == 0: 
            status = "✅ DONE" if is_success else "❌ FAIL"
            # Cập nhật in ấn để phản ánh EPSILON_DECAY_RATE mới
            print(f"Episode: {episode+1:6d} | Epsilon: {epsilon:.6f} | Steps: {env.steps:3d} | Reward: {total_reward:6.2f} | Status: {status} | Success Rate (last 1000): {success_count/1000:.2f} | States: {len(Q)}")
            success_count = 0 # Reset đếm thành công
        
        if (episode + 1) % 10000 == 0 or episode == EPISODES - 1:
            # Lưu Q-table và epsilon, episode
            with open(q_file, "wb") as f:
                pickle.dump({'Q': dict(Q), 'epsilon': epsilon, 'episode': episode}, f)
            print(f"💾 Đã lưu model tại episode {episode+1}. Kích thước Q-table: {len(Q)}")

    end_time = datetime.now()
    print(f"\n--- Hoàn thành huấn luyện {EPISODES} episodes ---")
    print(f"Tổng thời gian huấn luyện: {end_time - start_time}")

if __name__ == "__main__":
    print("Bắt đầu huấn luyện Q-learning (State Abstraction) với Tham số Tối ưu...")
    print(f"GAMMA={GAMMA}, ALPHA={ALPHA}, EPSILON_DECAY={EPSILON_DECAY_RATE}")
    train_qlearning_with_abstraction()
