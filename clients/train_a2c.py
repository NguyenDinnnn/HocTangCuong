import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv

# ========================
# Actor-Critic Model cho 5 kênh
# ========================
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
        conv_out_size = 32 * height * width  # flatten output
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
        return self.actor(features), self.critic(features)


# ========================
# Training Loop
# ========================
def train_a2c(episodes=500, gamma=0.99, lr=1e-3):
    env = GridWorldEnv(width=5, height=5, start=(0, 0), goal=(4, 4))
    in_channels = 5
    height, width = env.height, env.width
    n_actions = len(env.ACTIONS)

    model = ActorCritic(in_channels, height, width, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(episodes):
        state = env.reset()
        state_tensor = env.build_grid_state().unsqueeze(0)  # [1,5,H,W]

        log_probs, values, rewards = [], [], []
        done = False
        total_reward = 0

        while not done:
            logits, value = model(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))

            state_tensor = env.build_grid_state().unsqueeze(0)

        # ======= Advantage Update =======
        Qval = 0
        values.append(torch.tensor([[0.0]]))  # bootstrap
        returns = []
        for r in reversed(rewards):
            Qval = r + gamma * Qval
            returns.insert(0, Qval)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values[:-1])
        returns = torch.cat(returns).detach()

        advantage = returns - values.squeeze()

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward}")

    # ========================
    # Save model vào models/
    # ========================
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "a2c_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Training complete, model saved to {model_path}")


if __name__ == "__main__":
    train_a2c()
