import numpy as np
import sys, os, pickle, random
from collections import defaultdict 
import math 

# Thêm thư mục gốc vào path để import GridWorldEnv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv

# ---------------------------
# REWARD PARAMETERS (ĐÃ TỐI ƯU HÓA: Phạt nặng hơn để ưu tiên đường ngắn)
# ---------------------------
REWARD_PARAMS = {
    "step_penalty": -1.0, 
    "wall_penalty": -50,         # Tăng mạnh penalty
    "obstacle_penalty": -100,      # Tăng mạnh penalty
    "waypoint_reward": 50,
    "goal_reward": 200,
    "goal_before_waypoints_penalty": -50,
}

# ---------------------------
# Env + RL params
# ---------------------------
# Khởi tạo env cơ bản
width, height = 10, 8
env = GridWorldEnv(width=width, height=height, start=(0,0), goal=(9,7), 
                   obstacles=[(1,1),(2,3),(4,4),(5,1),(7,6)], 
                   waypoints=[(3,2),(6,5)], 
                   max_steps=500) 

# Áp dụng tham số Reward tối ưu vào môi trường
for k, v in REWARD_PARAMS.items():
    setattr(env, k, v)

# Q-Learning parameters
actions = ['up', 'right', 'down', 'left']
gamma = 0.995 # << TỐI ƯU HÓA: Tăng chiết khấu để ưu tiên phần thưởng dài hạn
alpha = 0.1
epsilon = 1.0
epsilon_min = 0.01 
epsilon_decay = 0.998 # Điều chỉnh nhẹ để kéo dài thời gian Exploration hơn, phù hợp với gamma cao
episodes = 50000 # Tăng số lượng episodes để đảm bảo hội tụ với gamma cao hơn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(BASE_DIR, exist_ok=True)
q_path = os.path.join(BASE_DIR, "qlearning_qtable_abstract_optimized.pkl") # Tên file mới

def _new_qrow():
    return {a: 0.0 for a in actions}

def encode_visited(wp_list, visited_set):
    """Mã hóa trạng thái Waypoint đã thăm dưới dạng một số nguyên."""
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

# >> HÀM TỐI ƯU HÓA TRẠNG THÁI TRỪU TƯỢNG (Enhanced State Abstraction)
def get_abstract_state(env):
    """
    Trả về trạng thái trừu tượng hóa (relative direction + local obstacles).
    Keys: (sign_dx_goal, sign_dy_goal, sign_dx_nearest_wp, sign_dy_nearest_wp, 
            visited_code, local_obstacle_code)
    """
    robot_x, robot_y = env.get_state()
    goal_x, goal_y = env.goal
    
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
    targets = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
    
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
    visited_code = encode_visited(env.waypoints, env.visited_waypoints)
    
    # TỐI ƯU HÓA 2: Local Obstacle Code (4-bit code)
    # Lên (0b0001), Phải (0b0010), Xuống (0b0100), Trái (0b1000)
    local_obstacle_code = 0
    
    # Vị trí các ô lân cận tương ứng với action: up, right, down, left
    neighbors = {
        'up': (robot_x, robot_y - 1),
        'right': (robot_x + 1, robot_y),
        'down': (robot_x, robot_y + 1),
        'left': (robot_x - 1, robot_y),
    }

    # Kiểm tra xem ô lân cận có phải là chướng ngại vật hay tường không
    if neighbors['up'] in env.obstacles or not (0 <= neighbors['up'][1] < env.height):
        local_obstacle_code |= 0b0001
    if neighbors['right'] in env.obstacles or not (0 <= neighbors['right'][0] < env.width):
        local_obstacle_code |= 0b0010
    if neighbors['down'] in env.obstacles or not (0 <= neighbors['down'][1] < env.height):
        local_obstacle_code |= 0b0100
    if neighbors['left'] in env.obstacles or not (0 <= neighbors['left'][0] < env.width):
        local_obstacle_code |= 0b1000
    
    # Kết hợp các đặc trưng tương đối thành Full Abstract State Key
    return (sign_dx_goal, sign_dy_goal, 
            sign_dx_nearest_wp, sign_dy_nearest_wp, 
            visited_code, local_obstacle_code)

# >> HÀM NGẪU NHIÊN HÓA MÔI TRƯỜNG (Domain Randomization)
def randomize_environment(env_instance, max_obstacles=10, max_waypoints=3):
    """
    Thiết lập ngẫu nhiên số lượng và vị trí Goal, Waypoints và Obstacles cho môi trường.
    Giữ nguyên logic randomization.
    """
    w, h = env_instance.width, env_instance.height
    start = env_instance.start 
    
    all_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) != start]
    random.shuffle(all_cells)
    
    # Ngẫu nhiên hóa số lượng Waypoints và Obstacles
    num_waypoints = random.randint(1, max_waypoints) 
    
    min_obs = 5 # Đảm bảo có một lượng chướng ngại vật nhất định
    max_cells_for_obs = len(all_cells) - num_waypoints - 1 
    
    num_obstacles = random.randint(min(min_obs, max_cells_for_obs), min(max_obstacles, max_cells_for_obs))
    
    num_reserved = num_waypoints + 1 # 1 cho goal, num_waypoints cho waypoints
    
    # 1. Random Obstacles
    # Đảm bảo không vượt quá số ô còn lại sau khi trừ đi start và các ô reserved
    if len(all_cells) - num_reserved < num_obstacles: 
        num_obstacles = len(all_cells) - num_reserved
    
    obstacles = all_cells[:num_obstacles]
    
    # 2. Random Waypoints và Goal
    remain = [cell for cell in all_cells if cell not in obstacles]
    
    # Đảm bảo có đủ ô trống cho Waypoints và Goal
    if len(remain) < num_reserved:
        needed = num_reserved - len(remain)
        if needed > 0:
            remain.extend(obstacles[-needed:]) # Lấy lại bớt obstacle nếu thiếu
            obstacles = obstacles[:-needed]
    
    # Chọn ngẫu nhiên Waypoints và Goal từ các ô còn lại
    if len(remain) >= num_reserved:
        chosen_coords = random.sample(remain, num_reserved)
    else:
        # Dùng toàn bộ ô còn lại nếu không đủ (trường hợp cực hiếm)
        chosen_coords = remain
        num_waypoints = len(chosen_coords) - 1 # Điều chỉnh số lượng waypoint
    
    if num_waypoints > 0:
        waypoints = chosen_coords[:-1]
        goal = chosen_coords[-1]
    elif chosen_coords: # Chỉ có goal
        waypoints = []
        goal = chosen_coords[0]
    else: # Trường hợp lỗi không còn ô nào
        goal = (w-1, h-1)
        waypoints = []
        obstacles = []


    # Cập nhật môi trường
    env_instance.obstacles = set(obstacles)
    env_instance.waypoints = waypoints
    env_instance.goal = goal


def _load_qtable():
    if os.path.exists(q_path) and os.path.getsize(q_path) > 0:
        try:
            with open(q_path, "rb") as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, dict) and 'Q' in loaded_data:
                return loaded_data['Q'], loaded_data.get('epsilon', epsilon)
            else:
                return loaded_data, epsilon
        except Exception as e:
            print(f"⚠️ Không load được {q_path}: {e}. Dùng bảng rỗng.")
    return {}, epsilon

# Q là defaultdict để tự động khởi tạo Q-value cho trạng thái mới
Q, current_epsilon = _load_qtable()
Q = defaultdict(_new_qrow, Q)
print(f"✅ Loaded Q-learning Q-table: {len(Q)} states. Epsilon: {current_epsilon:.4f}")


def choose_action(abstract_state, current_epsilon):
    """Chọn hành động theo chiến lược epsilon-greedy, sử dụng Abstract State."""
    if abstract_state not in Q: 
        Q[abstract_state] = _new_qrow()
    
    if random.random() < current_epsilon:
        return random.choice(actions) 
    else:
        q_values = Q[abstract_state] 
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

for ep in range(episodes):
    # I. BƯỚC QUAN TRỌNG: NGẪU NHIÊN HÓA MÔI TRƯỜNG
    randomize_environment(env, max_obstacles=10, max_waypoints=3) 
    
    # Reset env: Tăng Max Steps để robot không bị time-out trong môi trường phức tạp
    env.reset(max_steps=500) 
    
    abstract_state = get_abstract_state(env) # Dùng hàm Sign-based TỐI ƯU

    done = False
    episode_reward = 0
    
    while not done:
        # Chọn action theo abstract_state
        action = choose_action(abstract_state, current_epsilon)
        action_idx = actions.index(action)

        old_abstract_state = abstract_state 
        
        # Thực hiện action
        _, reward, done_raw, _ = env.step(action_idx)
        
        # Tạo next_abstract_state
        next_abstract_state = get_abstract_state(env) 

        episode_reward += reward

        # Kích hoạt done khi đạt mục tiêu hoặc hết max_steps
        done = env.is_done() or env.steps >= env.max_steps
        
        # Standard Q-learning update
        max_next_Q = 0.0
        if not done:
            # Lấy Q-value lớn nhất của trạng thái tiếp theo
            max_next_Q = max(Q[next_abstract_state].values())
        
        target = reward + gamma * max_next_Q
        
        # Cập nhật Q-value
        Q[old_abstract_state][action] += alpha * (target - Q[old_abstract_state][action])

        abstract_state = next_abstract_state
    
    # 3. Giảm epsilon
    current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
    
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep + 1}/{episodes} | Epsilon: {current_epsilon:.4f} | Steps: {env.steps} | Total Reward: {episode_reward:.2f} | States: {len(Q)}")

    if (ep + 1) % 500 == 0:
        # Lưu Q-table và epsilon định kỳ
        with open(q_path, "wb") as f:
            pickle.dump({'Q': dict(Q), 'epsilon': current_epsilon}, f)
        print(f"💾 Saved checkpoint at episode {ep + 1}.")

# Lưu Q-table cuối cùng
with open(q_path, "wb") as f:
    pickle.dump({'Q': dict(Q), 'epsilon': current_epsilon}, f)
print(f"\n✅ Training complete. Final Q-table saved to {q_path}.")
