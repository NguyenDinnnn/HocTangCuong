# clients/train_qlearning.py
import numpy as np
import sys, os, pickle, random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv

# Simplified but effective reward params
REWARD_PARAMS = {
    "step_penalty": -0.01,     # Very small step penalty
    "wall_penalty": -1,        # Moderate wall penalty  
    "obstacle_penalty": -2,    # Moderate obstacle penalty
    "waypoint_reward": 10,     # Reasonable waypoint reward
    "goal_reward": 50,         # High goal reward
    "goal_before_waypoints_penalty": -10,  # Penalty for wrong goal timing
}

# ---------------------------
# Env + RL params
# ---------------------------
env = GridWorldEnv(max_steps=500)
for k, v in REWARD_PARAMS.items():
    setattr(env, k, v)

# Simple and effective Q-Learning parameters
actions = ['up', 'right', 'down', 'left']
gamma = 0.9          # Good discount factor
alpha = 0.1          # Standard learning rate
epsilon = 1.0        # Start with full exploration
epsilon_min = 0.01   # Very low minimum for strong exploitation
epsilon_decay = 0.995  # Gradual decay
episodes = 500       # Fewer episodes, more focused training

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(BASE_DIR, exist_ok=True)
q_path = os.path.join(BASE_DIR, "qlearning_qtable.pkl")

def _new_qrow():
    return {a: 0.0 for a in actions}

def _load_qtable():
    if os.path.exists(q_path):
        try:
            with open(q_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Không load được {q_path}: {e}. Dùng bảng rỗng.")
    return {}

def _save_and_reload_q(Q):
    with open(q_path, "wb") as f:
        pickle.dump(Q, f)
    with open(q_path, "rb") as f:
        return pickle.load(f)

Q = _load_qtable()
print(f"✅ Loaded Q-learning Q-table: {len(Q)} states" if Q else "ℹ️ Start with empty Q-table")

def choose_action(state, current_epsilon):
    if state not in Q:
        Q[state] = _new_qrow()
    
    # Simple epsilon-greedy with Manhattan distance preference for exploration
    if random.random() < current_epsilon:
        # Exploration: prefer actions that move towards goal
        x, y = state
        goal_x, goal_y = env.goal
        
        valid_actions = []
        action_distances = []
        
        for action in actions:
            action_idx = actions.index(action)
            dx, dy = env.ACTIONS[action_idx]
            new_x, new_y = x + dx, y + dy
            
            # Check if action is valid
            if (0 <= new_x < env.width and 0 <= new_y < env.height and 
                (new_x, new_y) not in env.obstacles):
                valid_actions.append(action)
                distance = abs(new_x - goal_x) + abs(new_y - goal_y)
                action_distances.append(distance)
        
        if valid_actions:
            # 70% chance choose closest to goal, 30% random
            if random.random() < 0.7 and action_distances:
                best_idx = action_distances.index(min(action_distances))
                return valid_actions[best_idx]
            else:
                return random.choice(valid_actions)
        else:
            return random.choice(actions)
    else:
        # Exploitation: choose best Q-value
        return max(Q[state], key=Q[state].get)

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

def _apply_goal_guard_and_unified_reward(s, r):
    reached_goal_now = (tuple(s) == env.goal)
    if reached_goal_now and len(env.visited_waypoints) < len(env.waypoints):
        r += REWARD_PARAMS["goal_before_waypoints_penalty"]
    return r

current_epsilon = epsilon

for ep in range(episodes):
    state = tuple(env.reset(max_steps=200))  # Shorter episodes
    for k, v in REWARD_PARAMS.items():
        setattr(env, k, v)

    done = False
    episode_reward = 0
    
    while not done:
        action = choose_action(state, current_epsilon)
        action_idx = actions.index(action)

        old_state = state
        next_state, reward, done_raw, _ = env.step(action_idx)
        
        # Apply simple reward shaping
        reward = _apply_simple_reward_shaping(old_state, tuple(next_state), reward)
        reward = _apply_goal_guard_and_unified_reward(next_state, reward)
        episode_reward += reward
        
        done = (tuple(next_state) == env.goal and len(env.visited_waypoints) == len(env.waypoints)) or env.steps >= 200

        next_state = tuple(next_state)
        if next_state not in Q:
            Q[next_state] = _new_qrow()

        # Standard Q-learning update
        Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])

        state = next_state
    
    # Simple epsilon decay
    current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
    
    if ep % 100 == 0:
        print(f"Episode {ep}: Epsilon={current_epsilon:.3f}, Reward={episode_reward:.1f}, Steps={env.steps}")

# Save final Q-table
with open(q_path, "wb") as f:
    pickle.dump(Q, f)

print(f"Q-learning training xong! Saved at {q_path}")
