
# clients/random_agent.py
import requests, time, random

BASE = "http://127.0.0.1:8000"

def run_episode(render=False):
    # Reset
    r = requests.post(f"{BASE}/reset", json={})
    r.raise_for_status()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action = random.randint(0,3)
        s = requests.post(f"{BASE}/step", json={"action": action}).json()
        total_reward += s["reward"]
        done = s["done"]
        steps = s["steps"]
        if render:
            print(s["ascii"], f"\nAction={action}, Reward={s['reward']}, Steps={steps}\n")
            time.sleep(0.05)
    return total_reward, steps

if __name__ == "__main__":
    episodes = 5
    for ep in range(episodes):
        ep_reward, ep_steps = run_episode(render=False)
        print(f"[Random] Episode {ep+1}: reward={ep_reward}, steps={ep_steps}")
