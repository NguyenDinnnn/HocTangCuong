import numpy as np
import sys, os
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.robot_env import GridWorldEnv
import random
import pickle

# ---------------------------
# Cài đặt môi trường và tham số
# ---------------------------
# Sử dụng môi trường có waypoint để huấn luyện cho bài toán phức tạp hơn
start = (0,0)
goal = (9,7)
waypoints = [(3,2),(6,5)]
obstacles = [(1,1),(2,3),(4,4),(5,1),(7,6)]
env = GridWorldEnv(width=10, height=8, start=start, goal=goal, obstacles=obstacles, waypoints=waypoints, max_steps=500)

actions = ['up', 'right', 'down', 'left']
gamma = 0.99      # Hệ số chiết khấu
alpha = 0.1       # Tỷ lệ học
epsilon = 1.0     # Tỷ lệ khám phá ban đầu
epsilon_decay = 0.9995 # Tỷ lệ giảm epsilon
min_epsilon = 0.05    # Epsilon tối thiểu
episodes = 10000  # Tăng số lượng episodes để học tốt hơn

# ---------------------------
# Chuẩn bị Q-table và đường dẫn
# ---------------------------
# Tạo folder models nếu chưa có
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(models_dir, exist_ok=True)
q_path = os.path.join(models_dir, "sarsa_qtable.pkl")

# Load Q-table nếu đã tồn tại, nếu không thì tạo mới
if os.path.exists(q_path):
    with open(q_path, "rb") as f:
        # Sử dụng defaultdict để không bị lỗi Key-Error
        loaded_q = pickle.load(f)
        Q = defaultdict(lambda: {a: 0.0 for a in actions})
        Q.update(loaded_q)
    print(f"✅ Đã tải Q-table của SARSA, tổng số state đã biết = {len(Q)}")
else:
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    print("🆕 Không tìm thấy Q-table, tạo mới.")

# ---------------------------
# Hàm và Vòng lặp huấn luyện
# ---------------------------

def encode_visited(wp_list, visited_set):
    """Mã hóa các waypoints đã đi qua thành một số nguyên."""
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

def choose_action(state, current_epsilon):
    """Chọn hành động bằng chính sách epsilon-greedy."""
    if random.random() < current_epsilon:
        return random.choice(actions)
    # Nếu có nhiều hành động cùng có giá trị max, chọn ngẫu nhiên một trong số chúng
    max_q = max(Q[state].values())
    best_actions = [a for a, q in Q[state].items() if q == max_q]
    return random.choice(best_actions)

print(f"🚀 Bắt đầu huấn luyện SARSA với {episodes} episodes...")

for ep in range(episodes):
    env.reset()
    done = False
    
    # Lấy trạng thái và hành động ban đầu
    state_xy = env.get_state()
    visited_code = encode_visited(env.waypoints, env.visited_waypoints)
    state = (state_xy[0], state_xy[1], visited_code)
    
    action = choose_action(state, epsilon)
    
    while not done and env.steps < env.max_steps:
        # Thực hiện hành động
        action_idx = actions.index(action)
        next_state_xy, reward, done, _ = env.step(action_idx)
        
        # Lấy trạng thái và hành động tiếp theo
        next_visited_code = encode_visited(env.waypoints, env.visited_waypoints)
        next_state = (next_state_xy[0], next_state_xy[1], next_visited_code)
        
        next_action = choose_action(next_state, epsilon)
        
        # Cập nhật Q-value theo công thức SARSA
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
        # Di chuyển đến trạng thái và hành động tiếp theo
        state = next_state
        action = next_action

    # Giảm epsilon sau mỗi episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (ep + 1) % 100 == 0:
        print(f"-> Episode {ep + 1}/{episodes} | Epsilon: {epsilon:.4f}")

# ---------------------------
# Lưu kết quả
# ---------------------------
with open(q_path, "wb") as f:
    pickle.dump(dict(Q), f) # Chuyển về dict thường trước khi lưu
print(f"🎉 Huấn luyện SARSA hoàn tất! Đã lưu Q-table tại {q_path}")