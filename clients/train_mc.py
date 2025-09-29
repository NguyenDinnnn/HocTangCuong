import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
from app.robot_env import GridWorldEnv
import random
import pickle

env = GridWorldEnv()
width, height = 10, 8
start, goal = (0,0), (9,7)
waypoints = [(3,2),(6,5)]
obstacles = [(1,1),(2,3),(4,4),(5,1),(7,6)]
actions = ['up', 'right', 'down', 'left']
gamma = 0.9
epsilon = 0.1
episodes = 50000
EPSILON_START, EPSILON_MIN, EPSILON_DECAY = 1.0, 0.1, 0.995
env = GridWorldEnv(width,height,start,goal,obstacles,waypoints,max_steps=500)


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
    print(f"✅ Loaded existing MC Q-table, tổng state = {len(Q)}")
else:
    Q = {}

def encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1<<i)
    return code

def choose_action(state, epsilon):
    if np.random.rand() > epsilon:
        if state in mc_Q and any(mc_Q[state].values()):
            return max(mc_Q[state], key=mc_Q[state].get)
        else:
            unvisited = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
            target = unvisited[0] if unvisited else env.goal
            dx, dy = target[0]-state[0], target[1]-state[1]
            return 'right' if abs(dx)>abs(dy) and dx>0 else \
                   'left' if abs(dx)>abs(dy) else \
                   'down' if dy>0 else 'up'
    else:
        return np.random.choice(actions)
def compute_reward(next_state):
    r = -1.0
    if next_state in env.waypoints and next_state not in env.visited_waypoints:
        r += 5
    if next_state==env.goal and len(env.visited_waypoints)==len(env.waypoints):
        r += 10
    return r

for ep in range(episodes):
    state_xy = env.reset()
    env.episode_buffer = []
    done = False


    while not done:
        visited_code = encode_visited(env.waypoints, env.visited_waypoints)
        full_state = (state_xy[0], state_xy[1], visited_code)

        action_name = choose_action(full_state, epsilon)
        action_idx = actions.index(action_name)

        next_state, _, done, _ = env.step(action_idx)
        reward = compute_reward(next_state)
        env.episode_buffer.append((full_state, action_name, reward))
        state_xy = next_state
        done = done or env.steps >= env.max_steps or env.state == env.goal
        
        if done or env.steps>=env.max_steps: break
    visited_pairs = set()
    for t, (s,a,r) in enumerate(env.episode_buffer):
        G = sum((gamma**(k-t))*env.episode_buffer[k][2] for k in range(t,len(env.episode_buffer)))
        if (s,a) not in visited_pairs:
            mc_Q[s][a] += 0.1  *(G - mc_Q[s][a])
            visited_pairs.add((s,a))

    epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)
    
    if (ep+1)%1000==0:
        with open(q_path,'wb') as f:
            pickle.dump({'Q':dict(mc_Q),'epsilon':epsilon},f)
        print(f"Episode {ep+1}: epsilon={epsilon:.4f}, states={len(mc_Q)}")
# Lưu Q-table
with open(q_path, "wb") as f:
    pickle.dump(Q, f)
print(f"MC training xong! Saved at {q_path}")
