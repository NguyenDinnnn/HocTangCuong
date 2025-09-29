import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import sys, os

# --- THIẾT LẬP PATH ĐỂ IMPORT GridWorldEnv ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv

# --- CÁC THAM SỐ HYPERPARAMETERS ---
GAMMA = 0.99           # Hệ số chiết khấu
LR = 0.0007            # Tốc độ học
ENTROPY_BETA = 0.01    # Hệ số cho Entropy Loss
EPISODES = 50000       # Số tập huấn luyện mặc định


# ----------------------------------------------------
# 1. ĐỊNH NGHĨA MÔ HÌNH CNN A2C (Actor-Critic)
# ----------------------------------------------------
class ActorCritic(nn.Module):
    """
    Mô hình A2C sử dụng CNN cho đầu vào tensor 5 kênh (5, H, W).
    """
    def __init__(self, in_channels, height, width, n_actions): 
        super().__init__()
        
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        feature_size = 64 * height * width

        # Actor trả về logits (KHÔNG softmax ở đây)
        self.actor = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        shared_output = self.cnn_extractor(x)
        policy_logits = self.actor(shared_output)
        value = self.critic(shared_output)
        
        return policy_logits, value


# ----------------------------------------------------
# 2. HÀM HUẤN LUYỆN CHÍNH
# ----------------------------------------------------
def calculate_returns(rewards: list, last_value: float):
    """Tính toán Lợi nhuận Chiết khấu (Target Return) từ chuỗi phần thưởng."""
    returns = []
    R = last_value
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    return torch.tensor(returns).float()


def train_a2c(env, model: ActorCritic, num_episodes: int):
    """
    Thực hiện thuật toán huấn luyện Advantage Actor-Critic (A2C).
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    episode_rewards = []
    
    print(f"Bắt đầu huấn luyện A2C trên môi trường {env.width}x{env.height} với {num_episodes} tập...")

    for episode in range(num_episodes):
        
        env.reset()
        state_tensor = env.build_grid_state() 
        done = False
        
        log_probs, values, rewards, entropy_terms = [], [], [], []
        total_reward = 0

        while not done and env.steps < env.max_steps:
            state_tensor_batch = state_tensor.unsqueeze(0)
            
            policy_logits, value = model(state_tensor_batch)
            action_probs = torch.softmax(policy_logits, dim=-1)  # softmax ở đây
            m = Categorical(action_probs)
            action = m.sample()
            
            _, reward, done, _ = env.step(action.item())
            next_state_tensor = env.build_grid_state()
            
            log_probs.append(m.log_prob(action))
            values.append(value)
            rewards.append(reward)
            entropy_terms.append(m.entropy())
            total_reward += reward
            
            state_tensor = next_state_tensor

        episode_rewards.append(total_reward)

        if done:
            last_value = 0.0
        else:
            state_tensor_batch = state_tensor.unsqueeze(0)
            with torch.no_grad():
                _, last_value_tensor = model(state_tensor_batch)
                last_value = last_value_tensor.item()
            
        returns = calculate_returns(rewards, last_value)
        
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        entropy_terms = torch.cat(entropy_terms)
        
        advantages = returns - values
        
        critic_loss = advantages.pow(2).mean() 
        actor_loss = -(log_probs * advantages.detach()).mean() 
        entropy_loss = -ENTROPY_BETA * entropy_terms.mean()

        total_loss = actor_loss + critic_loss + entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {episode + 1}/{num_episodes} | Avg Reward (100eps): {avg_reward:.2f} | Loss: {total_loss.item():.4f}")
            
    print("Quá trình huấn luyện A2C hoàn tất.")
    return model, episode_rewards


# ----------------------------------------------------
# 3. CHẠY CHƯƠNG TRÌNH CHÍNH (MAIN BLOCK)
# ----------------------------------------------------
if __name__ == "__main__":
    
    print("--- Chế độ chạy thử nghiệm A2C cục bộ ---")
    
    TEST_WIDTH = 10
    TEST_HEIGHT = 8
    TEST_START = (0,0)
    TEST_GOAL = (9,7)
    TEST_WAYPOINTS = [(3, 2), (6, 5)]
    TEST_OBSTACLES = [(1,1),(2,3),(4,4),(5,1),(7,6)]
    TEST_MAX_STEPS = 500
    
    # Khởi tạo môi trường
    test_env = GridWorldEnv(
        width=TEST_WIDTH, 
        height=TEST_HEIGHT, 
        start=TEST_START, 
        goal=TEST_GOAL,
        obstacles=TEST_OBSTACLES,
        waypoints=TEST_WAYPOINTS,
        max_steps=TEST_MAX_STEPS
    )
    
    ACTION_SIZE = len(test_env.ACTIONS)

    # Khởi tạo mô hình A2C
    a2c_model = ActorCritic(in_channels=5, height=TEST_HEIGHT, width=TEST_WIDTH, n_actions=ACTION_SIZE)

    # Bắt đầu huấn luyện
    final_model, rewards_history = train_a2c(test_env, a2c_model, num_episodes=EPISODES)
    
    print("\n--- Kết quả sau huấn luyện ---")
    print(f"Phần thưởng trung bình 100 tập cuối: {np.mean(rewards_history[-100:]):.2f}")
