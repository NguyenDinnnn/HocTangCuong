import os
import sys
import random
import pickle
import time
from fastapi import FastAPI
from pydantic import BaseModel
from threading import Lock
from contextlib import asynccontextmanager

# Thêm đường dẫn tới app (Giả định cấu trúc thư mục)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.robot_env import GridWorldEnv

# ---------------------
# 1. HYPERPARAMETERS
# ---------------------
actions = ['up', 'right', 'down', 'left']
gamma = 0.99
alpha = 0.2
epsilon_init = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9998
episodes_default = 50000
TRAIN_MAX_STEPS = 400

# ---------------------
# 2. Q-TABLE
# ---------------------
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)
ql_qfile = os.path.join(models_dir, "qlearning_qtable.pkl")
ql_Q = {}

if os.path.exists(ql_qfile):
    with open(ql_qfile, "rb") as f:
        ql_Q = pickle.load(f)
    print("✅ Loaded Q-table cũ.")
else:
    print("⚠️ Chưa có Q-table, tạo mới.")

# ---------------------
# 3. HELPER
# ---------------------
def encode_visited(waypoints, visited):
    code = 0
    for i, wp in enumerate(waypoints):
        if wp in visited:
            code |= (1 << i)
    return code

def choose_action(state, epsilon):
    global ql_Q
    if state not in ql_Q:
        ql_Q[state] = {a: 0.0 for a in actions}
    if random.random() < epsilon:
        return random.choice(actions)
    max_q = max(ql_Q[state].values())
    best_actions = [a for a, q in ql_Q[state].items() if q == max_q]
    return random.choice(best_actions)

# ---------------------
# 4. FASTAPI SETUP
# ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.env_lock = Lock()
    yield

app = FastAPI(lifespan=lifespan)

class TrainRequest(BaseModel):
    episodes: int = episodes_default

# ---------------------
# 5. TRAIN + RUN
# ---------------------
def train_and_run_qlearning(req: TrainRequest, env_lock=None):
    global ql_Q, alpha, gamma, epsilon_init, epsilon_decay, epsilon_min

    if env_lock is None:
        env_lock = Lock()  # tạo lock nếu chạy offline

    episodes = req.episodes
    epsilon = epsilon_init
    env_train = GridWorldEnv(width=10, height=8, max_steps=TRAIN_MAX_STEPS)

    with env_lock:
        # Train loop
        for ep in range(episodes):
            env_train.randomize_map(n_obstacles=5, n_waypoints=2)
            start = env_train.reset()
            env_train.visited_waypoints = set()
            remaining_waypoints = set(env_train.waypoints)
            visited_cells = set()
            state = (start[0], start[1], encode_visited(env_train.waypoints, env_train.visited_waypoints))
            done = False

            while not done and env_train.steps < env_train.max_steps:
                action = choose_action(state, epsilon)
                action_idx = actions.index(action)
                prev_x, prev_y = state[0], state[1]

                target = min(remaining_waypoints, key=lambda w: abs(prev_x - w[0]) + abs(prev_y - w[1])) \
                         if remaining_waypoints else env_train.goal

                next_xy, reward_base, done, _ = env_train.step(action_idx)
                old_dist = abs(prev_x - target[0]) + abs(prev_y - target[1])
                new_dist = abs(next_xy[0] - target[0]) + abs(next_xy[1] - target[1])
                reward = reward_base
                if next_xy == target:
                    reward += 15.0
                elif new_dist < old_dist:
                    reward += 0.8
                elif new_dist > old_dist:
                    reward -= 0.5
                if next_xy in visited_cells:
                    reward -= 0.1
                if next_xy == (prev_x, prev_y):
                    reward -= 2.0
                visited_cells.add(next_xy)
                if next_xy in remaining_waypoints:
                    remaining_waypoints.remove(next_xy)
                    env_train.visited_waypoints.add(next_xy)

                next_state = (next_xy[0], next_xy[1], encode_visited(env_train.waypoints, env_train.visited_waypoints))
                if next_state not in ql_Q:
                    ql_Q[next_state] = {a: 0.0 for a in actions}

                ql_Q[state][action] += alpha * (
                    reward + gamma * max(ql_Q[next_state].values()) - ql_Q[state][action]
                )
                state = next_state

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        with open(ql_qfile, "wb") as f:
            pickle.dump(ql_Q, f)
        print("✅ Đã lưu Q-table sau khi train.")

        # Run greedy
        env_train.max_steps = 200
        env_train.randomize_map(n_obstacles=5, n_waypoints=2)
        start = env_train.reset()
        env_train.visited_waypoints = set()
        remaining_waypoints = set(env_train.waypoints)
        visited_cells = set()
        state = (start[0], start[1], encode_visited(env_train.waypoints, env_train.visited_waypoints))
        done = False
        total_reward = 0
        steps = 0
        rewards_over_time = []
        start_time = time.time()

        while not done and steps < env_train.max_steps:
            target = min(remaining_waypoints, key=lambda w: abs(state[0]-w[0])+abs(state[1]-w[1])) \
                     if remaining_waypoints else env_train.goal
            if state in ql_Q:
                max_q = max(ql_Q[state].values())
                best_actions = [a for a, q in ql_Q[state].items() if q == max_q]
                action = random.choice(best_actions)
            else:
                action = random.choice(actions)
            action_idx = actions.index(action)
            next_xy, reward_base, done, _ = env_train.step(action_idx)
            if next_xy in remaining_waypoints:
                remaining_waypoints.remove(next_xy)
                env_train.visited_waypoints.add(next_xy)
            state = (next_xy[0], next_xy[1], encode_visited(env_train.waypoints, env_train.visited_waypoints))
            total_reward += reward_base
            rewards_over_time.append(total_reward)
            steps += 1

        elapsed_time = time.time() - start_time

        return {
            "algorithm": "Q-Learning",
            "reward": total_reward,
            "steps": steps,
            "visited_waypoints": len(env_train.visited_waypoints),
            "elapsed_time": elapsed_time,
            "rewards_over_time": rewards_over_time,
            "ascii": env_train.render_ascii()
        }

# ---------------------
# 6. CHẠY TRỰC TIẾP
# ---------------------
if __name__ == "__main__":
    class DummyReq(BaseModel):
        episodes: int = 1000
    res = train_and_run_qlearning(DummyReq())
    print("=== Kết quả Run Greedy ===")
    print(f"Reward: {res['reward']}, Steps: {res['steps']}, Waypoints visited: {res['visited_waypoints']}")
    print(res['ascii'])
