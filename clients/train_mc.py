import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
from app.robot_env import GridWorldEnv
import random
import pickle

width, height = 10, 8
start, goal = (0,0), (9,7) 
waypoints = [(3,2),(6,5)]
obstacles = [(4,4),(1,1),(5,1),(2,3),(7,6)]
MAX_STEPS = 500

actions = ['up', 'right', 'down', 'left']
gamma = 0.99
episodes = 50000
EPSILON_START, EPSILON_MIN, EPSILON_DECAY = 1.0, 0.1, 0.99995

# KHỞI TẠO MÔI TRƯỜNG CỐ ĐỊNH BAN ĐẦU
env = GridWorldEnv(width,height,start,goal,obstacles,waypoints,max_steps=MAX_STEPS)


# Tạo folder models tuyệt đối
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(BASE_DIR, exist_ok=True)
q_path = os.path.join(BASE_DIR, "mc_qtable.pkl") 

mc_Q = defaultdict(lambda:{a:0.0 for a in actions})
epsilon = EPSILON_START

# Load Q-table nếu đã tồn tại
if os.path.exists(q_path) and os.path.getsize(q_path) > 0:
    
    with open(q_path, "rb") as f:
        Q = pickle.load(f)
        mc_Q.update(Q.get('Q',{}))
        epsilon = Q.get('epsilon', EPSILON_START)
    print(f"Đã tải MC Q-table, tổng state = {len(mc_Q)} và epsilon = {epsilon:.4f}")
else:
    print("Không tìm thấy MC Q-table, bắt đầu với Q/Epsilon mặc định.")
    Q = {}

def encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1<<i)
    return code

def choose_action(full_state, epsilon):
    if np.random.rand() > epsilon:
        # Khai thác (Exploit)
        if full_state in mc_Q and any(mc_Q[full_state].values()):
            return max(mc_Q[full_state], key=mc_Q[full_state].get)
        else:
            # Nếu trạng thái mới/rỗng, chọn ngẫu nhiên
            return np.random.choice(actions)
    else:
        # Khám phá (Explore)
        return np.random.choice(actions)
        
def get_full_state(env, state_xy):
    # State chỉ bao gồm vị trí (x, y) và các waypoint đã thăm
    visited_code = encode_visited(env.waypoints, env.visited_waypoints)
    return (state_xy[0], state_xy[1], visited_code)


for ep in range(episodes):
    # --- RESET MÔI TRƯỜNG CỐ ĐỊNH ---
    # env.reset() sẽ tự động sử dụng lại các tham số Goal/Obstacle/Waypoint cố định
    state_xy = env.reset()
    # ---------------------------------
    
    episode_buffer = [] 
    done = False

    while not done:
        # Lấy trạng thái trước khi hành động
        full_state = get_full_state(env, state_xy)

        action_name = choose_action(full_state, epsilon)
        action_idx = actions.index(action_name)

        # THỰC HIỆN STEP
        next_state, reward, done_step, _ = env.step(action_idx)
        
        # Lưu trữ (S, A, R)
        episode_buffer.append((full_state, action_name, reward))
        state_xy = next_state
        
        # Kiểm tra done 
        done = done_step or env.steps >= env.max_steps
        
        if done: break 

    # CẬP NHẬT MC (First-Visit)
    visited_pairs = set()
    T = len(episode_buffer)
    
    for t, (s, a, r) in enumerate(episode_buffer):
        if (s,a) in visited_pairs:
            continue
        
        # Tính Return G_t
        G = sum((gamma**(k-t)) * episode_buffer[k][2] for k in range(t, T))
        
        # Cập nhật Q-value
        mc_Q[s][a] += 0.1 * (G - mc_Q[s][a])
        visited_pairs.add((s,a))

    # GIẢM EPSILON
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if (ep + 1) % 1000 == 0:
        with open(q_path, 'wb') as f:
            pickle.dump({'Q': dict(mc_Q), 'epsilon': epsilon}, f)
        
        # Logging
        total_reward = sum(r for s, a, r in episode_buffer)
        print(f"Episode {ep+1}/{episodes}: Total Reward={total_reward:.4f}, Epsilon={epsilon:.4f}, States={len(mc_Q),} Steps={env.steps}")

# LƯU CUỐI CÙNG
with open(q_path, "wb") as f:
    pickle.dump({'Q': dict(mc_Q), 'epsilon': epsilon}, f)
print(f"MC training xong trên bản đồ cố định! Saved at {q_path}")