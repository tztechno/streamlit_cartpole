
import streamlit as st
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time
from matplotlib import animation
import matplotlib
import os
from PIL import Image
import io


matplotlib.use('Agg')  # バックエンドの設定

st.set_page_config(page_title="CartPole DQN", layout="wide")
st.title("CartPole DQN")

# Training parameters in sidebar
st.sidebar.header("Training Parameters")
gamma = st.sidebar.slider("Discount Factor (GAMMA)", min_value=0.8, max_value=0.999, value=0.99, step=0.01)
lr = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
epsilon_start = st.sidebar.slider("Initial Epsilon", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
epsilon_decay = st.sidebar.slider("Epsilon Decay", min_value=0.9, max_value=0.999, value=0.995, step=0.001)
epsilon_min = st.sidebar.slider("Minimum Epsilon", min_value=0.01, max_value=0.2, value=0.01, step=0.01)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
memory_size = st.sidebar.slider("Memory Size", min_value=1000, max_value=100000, value=10000, step=1000)
hidden_size = st.sidebar.slider("Hidden Layer Size", min_value=16, max_value=128, value=24, step=8)
target_update = st.sidebar.slider("Target Network Update (episodes)", min_value=1, max_value=50, value=10, step=1)
num_episodes = st.sidebar.slider("Number of Episodes", min_value=100, max_value=2000, value=500, step=100)

# Network architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Training function
def train_agent():
    # Progress info
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize networks
    policy_net = DQN(state_dim, action_dim, hidden_size)
    target_net = DQN(state_dim, action_dim, hidden_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    # Memory
    memory = ReplayMemory(memory_size)
    
    # Metrics tracking
    rewards_history = []
    epsilon_history = []
    loss_history = []
    epsilon = epsilon_start
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        step_count = 0
        
        for t in range(500):  # Max episode length
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state)).argmax().item()
            
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in memory
            memory.push(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Train network if enough samples
            if len(memory) >= batch_size:
                # Sample batch
                transitions = memory.sample(batch_size)
                batch = list(zip(*transitions))
                
                # Prepare batch data
                states = torch.FloatTensor(batch[0])
                actions = torch.LongTensor(batch[1]).unsqueeze(1)
                rewards = torch.FloatTensor(batch[2])
                next_states = torch.FloatTensor(batch[3])
                dones = torch.FloatTensor(batch[4])
                
                # Compute Q values
                q_values = policy_net(states).gather(1, actions).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                # Compute loss and optimize
                loss = nn.MSELoss()(q_values, target_q_values.detach())
                episode_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        # Update metrics
        avg_loss = episode_loss / step_count if step_count > 0 else 0
        rewards_history.append(total_reward)
        epsilon_history.append(epsilon)
        loss_history.append(avg_loss)
        
        # Update progress
        progress_bar.progress((episode + 1) / num_episodes)
        status_text.text(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Steps: {step_count}, Epsilon: {epsilon:.3f}")
    
    env.close()
    
    # Return the trained policy network and history data
    return policy_net, rewards_history, epsilon_history, loss_history

def visualize_cartpole(policy_net, epsilon=0.05):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset()
    frames = []

    for t in range(500):
        frame = env.render()  # render_mode="rgb_array" でNumPy配列として取得
        frames.append(frame)

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state

        if terminated or truncated:
            break

    env.close()

    # PIL Imageに変換
    images = [Image.fromarray(frame) for frame in frames]

    # メモリ上にGIFを保存
    gif_buffer = io.BytesIO()
    images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=50,  # ミリ秒
        loop=0
    )
    gif_buffer.seek(0)
    
    return gif_buffer


def test_agent(policy_net, num_test_episodes=3):
    st.subheader("Agent Performance")
    
    # まずアニメーションを生成して表示
    with st.spinner("Generating animation..."):
        gif_buffer = visualize_cartpole(policy_net)
        st.image(gif_buffer, width=600, caption="CartPole Animation")
    
    # テスト用環境作成
    env = gym.make("CartPole-v1")
    
    # エージェントのテスト
    test_rewards = []
    test_steps = []
    
    for i in range(num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for t in range(500):  # 最大エピソード長
            # ポリシーを使用
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()
            
            # アクション実行
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        test_rewards.append(episode_reward)
        test_steps.append(steps)
    
    env.close()
    
    # テスト結果を表示
    avg_reward = sum(test_rewards) / len(test_rewards)
    avg_steps = sum(test_steps) / len(test_steps)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Reward", f"{avg_reward:.1f}")
    with col2:
        st.metric("Average Steps", f"{avg_steps:.1f}")
    
    # テスト結果を表にして表示
    results_df = {"Episode": list(range(1, num_test_episodes+1)),
                 "Reward": test_rewards,
                 "Steps": test_steps}
    
    st.write("Test Episode Results:")
    st.dataframe(results_df)
    
    return avg_reward

    

## Main script
st.markdown("""
This app trains a Deep Q-Network (DQN) agent to solve the CartPole environment 
from OpenAI Gymnasium. Adjust the parameters in the sidebar to see how they 
affect the training process.

### Task Description
In CartPole, a pole is attached to a cart moving along a track. The goal is to 
balance the pole by applying forces to the cart. A reward of +1 is provided for 
every timestep that the pole remains upright. The episode ends when the pole is 
more than 15 degrees from vertical, or the cart moves more than 2.4 units from 
the center.
""")

# 静止画を表示せずにプレースホルダーを配置
animation_placeholder = st.empty()

if st.button("Train Agent"):
    with st.spinner("Training DQN agent... This might take a few minutes."):
        start_time = time.time()
        policy_net, rewards_history, epsilon_history, loss_history = train_agent()
        training_time = time.time() - start_time
        st.success(f"Training completed in {training_time:.1f} seconds!")
        
        # Create final results visualization
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        
        # Rewards plot
        ax[0].plot(rewards_history)
        ax[0].set_title('Episode Rewards')
        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Total Reward')
        ax[0].grid(True)
        
        # Epsilon plot
        ax[1].plot(epsilon_history)
        ax[1].set_title('Epsilon Decay')
        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('Epsilon')
        ax[1].grid(True)
        
        # Loss plot
        ax[2].plot(loss_history)
        ax[2].set_title('Training Loss')
        ax[2].set_xlabel('Episode')
        ax[2].set_ylabel('Loss')
        ax[2].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Test the trained agent
        st.subheader("Testing Trained Agent")
        test_agent(policy_net)



