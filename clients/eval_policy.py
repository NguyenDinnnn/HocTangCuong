
# clients/eval_policy.py
# Evaluate a saved policy (mc_q.npy or qlearning_q.npy) by running greedy actions.
import requests, numpy as np, os, sys, time

BASE = "http://127.0.0.1:8000"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def get_map():
    r = requests.get(f"{BASE}/map").json()
    m = r["map"]
    return m["width"], m["height"]

def run_episode(Q, render=False):
    requests.post(f"{BASE}/reset", json={}).json()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        st = requests.get(f"{BASE}/state").json()
        x, y = st["state"]
        a = int(np.argmax(Q[y, x]))
        s = requests.post(f"{BASE}/step", json={"action": a}).json()
        total_reward += s["reward"]
        steps = s["steps"]
        done = s["done"]
        if render:
            print(s["ascii"], f"\nAction={a}, Reward={s['reward']}, Steps={steps}\n")
            time.sleep(0.05)
    return total_reward, steps

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("mc", "q"):
        print("Usage: python eval_policy.py [mc|q] [episodes=20] [render=0/1]")
        sys.exit(1)
    which = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    render = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    fname = "mc_q.npy" if which == "mc" else "qlearning_q.npy"
    path = os.path.join(MODEL_DIR, fname)
    Q = np.load(path)
    rewards = []
    steps_list = []
    for ep in range(episodes):
        r, s = run_episode(Q, render=render)
        rewards.append(r); steps_list.append(s)
    print(f"[Eval-{which}] Episodes={episodes} avg_reward={np.mean(rewards):.2f} avg_steps={np.mean(steps_list):.2f}")
