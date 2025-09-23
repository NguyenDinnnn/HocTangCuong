
# RL Robot Path Optimization (FastAPI + Console)

This project shows a full pipeline for robot path optimization using Reinforcement Learning (Monte Carlo and Q-learning),
with a **FastAPI server** hosting the environment and **console clients** that train/evaluate via HTTP.

## 🗂 Structure

```
rl_fastapi_robot/
├── app/
│   ├── robot_env.py      # GridWorld environment
│   └── server.py         # FastAPI server
├── clients/
│   ├── random_agent.py   # Baseline (random actions)
│   ├── train_mc.py       # Monte Carlo control training
│   ├── train_qlearning.py# Q-learning training
│   └── eval_policy.py    # Evaluate trained policies
├── models/               # Saved Q tables (.npy)
├── requirements.txt
└── README.md
```

## ✅ Step-by-step

### 0) Create venv & install dependencies
```bash
cd rl_fastapi_robot
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Start the FastAPI environment server
```bash
# from project root
uvicorn app.server:app --reload --port 8000
```
Open another terminal for clients.

- Test endpoints:
```bash
curl http://127.0.0.1:8000/map
curl -X POST http://127.0.0.1:8000/reset
curl -X POST http://127.0.0.1:8000/step -H "Content-Type: application/json" -d "{\"action\": 1}"
```

> Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.

### 2) Run baseline (random) to get the "pre-learning" behavior
```bash
python clients/random_agent.py
```

### 3) Train Monte Carlo (MC) policy
```bash
python clients/train_mc.py
# model saved to models/mc_q.npy
```

### 4) Train Temporal-Difference (TD) = Q-learning policy
```bash
python clients/train_qlearning.py
# model saved to models/qlearning_q.npy
```

### 5) Evaluate the trained policies (greedy)
```bash
python clients/eval_policy.py mc 20 0    # evaluate MC policy, 20 episodes, no render
python clients/eval_policy.py q  20 0    # evaluate Q-learning policy
# Add '1' at the end to render ASCII steps:
python clients/eval_policy.py q 5 1
```

## ⚙️ Customize the map
You can reset the environment with new map parameters:
```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"width":12,"height":12,"start":[0,0],"goal":[11,11],"obstacles":[[3,3],[3,4],[3,5],[6,6],[7,6],[6,7],[8,4]],"max_steps":300}'
```

## 📌 Notes
- The environment is a simplified 2D GridWorld to focus on RL pipeline. You can later parameterize obstacles/goal using your dataset or a learned cost map.
- Monte Carlo updates after full episode; Q-learning updates every step.
- Policies are tabular (Q[y, x, 4]) for clarity.
- Everything runs via HTTP so you can later swap the console client with a web UI easily.
```

