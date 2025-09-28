import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys, os
from torch.distributions import Categorical

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv

# =======================
# Actor-Critic Model cho 5 kênh
# =======================
class ActorCritic(nn.Module):
    def __init__(self, in_channels, height, width, n_actions):
        super().__init__()
        # Conv2d để xử lý grid 5 kênh
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = 32 * height * width
        hidden = 128
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.conv(x)
        features = self.fc(x)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value

# =======================
# Hàm Train A2C ĐÃ CHỈNH SỬA VÀ TỐI ƯU
# =======================
def train_a2c(model, env, optimizer, episodes=1000, gamma=0.95, max_steps=500):
    # CÁC THAM SỐ ỔN ĐỊNH THUẬT TOÁN
    CRITIC_COEFF = 0.5    # Giảm trọng số Critic Loss
    ENTROPY_COEFF = 0.01  # Khuyến khích khám phá
    REWARD_SCALE = 0.05   # Chia tỷ lệ Reward để ổn định tín hiệu
    
    for ep in range(episodes):
        env.reset()
        done = False
        total_reward = 0
        
        log_probs = []
        values = []
        rewards = []
        entropies = [] 
        
        state_tensor = env.build_grid_state().unsqueeze(0)

        # 2. Vòng lặp thu thập dữ liệu (Rollout)
        for step in range(max_steps):
            policy_logits, value = model(state_tensor)
            
            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.item())
            total_reward += reward

            # Lưu dữ liệu
            log_probs.append(dist.log_prob(action))
            values.append(value)
            
            # Áp dụng Reward Scaling
            scaled_reward = reward * REWARD_SCALE 
            rewards.append(torch.tensor([scaled_reward], dtype=torch.float32)) 
            
            entropies.append(dist.entropy())

            state_tensor = env.build_grid_state().unsqueeze(0)

            if done:
                break

        # ==========================================================
        # ======= Advantage Update (A2C CHUẨN - TD-BASED) =========
        # ==========================================================
        
        # 1. Tính V_final (Bootstrapping Value)
        last_state_tensor = state_tensor 
        
        if done:
            V_final = torch.tensor([0.0])
        else:
            with torch.no_grad(): 
                _, V_final_tensor = model(last_state_tensor)
            V_final = V_final_tensor.squeeze()

        # 2. Tính N-step Return (Target G_t)
        G = V_final.detach() 
        returns = []
        
        for r in reversed(rewards):
            G = r.squeeze() + gamma * G 
            returns.insert(0, G)
        
        # 3. Chuẩn bị cho Loss
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        returns = torch.stack(returns).detach() 

        # 4. Tính Advantage
        advantage = returns - values 

        # CHUẨN HÓA ADVANTAGE
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # 5. Tính Loss Function
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean() 
        entropy_loss = torch.cat(entropies).mean()
        
        # Loss tổng
        loss = actor_loss + (CRITIC_COEFF * critic_loss) - (ENTROPY_COEFF * entropy_loss)

        # 6. Cập nhật Model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            reached_waypoints = len(env.visited_waypoints)
            reached_goal = (env.state == env.goal)
            print(
                f"Episode {ep}, Total Reward: {total_reward}, "
                f"Waypoints reached: {reached_waypoints}/{len(env.waypoints)}, "
                f"Reached goal: {reached_goal}"
            )

    # ========================
    # Save model vào models/
    # ========================
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'a2c_model.pth'))
    print("A2C model saved successfully.")
    # =======================
# Main
# =======================
if __name__ == "__main__":
    env = GridWorldEnv(
        width=10, height=8,
        start=(0,0), goal=(9,7),
        obstacles=[(1,1),(2,3),(4,4),(5,1),(7,6)],
        waypoints=[(3,2),(6,5)],
        max_steps=300
    )

    in_channels = 5
    n_actions = len(env.ACTIONS)
    model = ActorCritic(in_channels, env.height, env.width, n_actions)

    model_path = os.path.join(os.path.dirname(__file__), "models", "a2c_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded previous A2C model, continue training...")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # CHỈ GỌI THAM SỐ ĐÚNG
    train_a2c(model, env, optimizer, episodes=1000)