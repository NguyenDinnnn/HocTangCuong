import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.robot_env import GridWorldEnv
import random
import pickle

env = GridWorldEnv()
actions = ['up', 'right', 'down', 'left']
gamma = 0.9
epsilon = 0.1
episodes = 5000

# Tạo folder models tuyệt đối
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(BASE_DIR, exist_ok=True)
q_path = os.path.join(BASE_DIR, "mc_qtable.pkl")

# Load Q-table nếu đã tồn tại
if os.path.exists(q_path):
    with open(q_path, "rb") as f:
        Q = pickle.load(f)
    print(f"✅ Loaded existing MC Q-table, tổng state = {len(Q)}")
else:
    Q = {}

def choose_action(state):
    if state not in Q:
        Q[state] = {a: 0.0 for a in actions}
    if random.random() < epsilon:
        return random.choice(actions)
    return max(Q[state], key=Q[state].get)

for ep in range(episodes):
    state = tuple(env.reset())
    episode = []
    done = False

    while not done:
        action = choose_action(state)
        action_idx = actions.index(action)
        next_state, reward, done, _ = env.step(action_idx)
        episode.append((state, action, reward))
        state = tuple(next_state)

    G = 0
    visited = set()
    for state, action, reward in reversed(episode):
        G = gamma * G + reward
        if (state, action) not in visited:
            if state not in Q:
                Q[state] = {a: 0.0 for a in actions}
            Q[state][action] += 0.1 * (G - Q[state][action])
            visited.add((state, action))

# Lưu Q-table
with open(q_path, "wb") as f:
    pickle.dump(Q, f)
print(f"MC training xong! Saved at {q_path}")
