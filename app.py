
import streamlit as st
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
import io
from PIL import Image

st.set_page_config(page_title="CartPole DQN", layout="wide")
st.title("CartPole DQN with Reinforcement Learning")

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

# Custom visualization of CartPole state
def draw_cart_pole(state, fig, ax):
    cart_x, cart_y = state[0], 0
    pole_length = 0.5
    theta = state[2]
    
    # Cart dimensions
    cart_width = 0.2
    cart_height = 0.1
    
    # Clear the axis
    ax.clear()
    
    # Draw the track
    ax.plot([-2.4, 2.4], [0, 0], 'k-', lw=1)
    
    # Draw the cart
    cart_left = cart_x - cart_width/2
    cart_right = cart_x + cart_width/2
    cart_bottom = cart_y - cart_height/2
    cart_top = cart_y + cart_height/2
    ax.add_patch(plt.Rectangle((cart_left, cart_bottom), cart_width, cart_height, fc='blue'))
    
    # Draw the pole
    pole_x = cart_x + pole_length * np.sin(theta)
    pole_y = cart_y + pole_length * np.cos(theta)
    ax.plot([cart_x, pole_x], [cart_y, pole_y], 'r-', lw=3)
    
    # Set axis limits and labels
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1.0)
    ax.set_aspect('equal')
    ax.set_title('CartPole Simulation')
    ax.set_xticks([])
    ax.set_yticks([])

# Training function
def train_agent():
    # Progress info
    progress_bar = st.progress(0)
    episode_info = st.empty()
    stats_text = st.empty()
    
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
        episode_info.info(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Steps: {step_count}, Epsilon: {epsilon:.3f}")
        
        # Update stats every 10 episodes
        if episode % 10 == 0 or episode == num_episodes - 1:
            avg_reward = sum(rewards_history[-10:]) / min(10, len(rewards_history[-10:]))
            stats_text.text(f"Last 10 Episodes - Avg Reward: {avg_reward:.1f}, Current ε: {epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
            
            # Plot metrics
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            # Rewards plot
            ax1.plot(rewards_history)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            
            # Epsilon plot
            ax2.plot(epsilon_history)
            ax2.set_title('Epsilon Decay')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            
            # Loss plot
            ax3.plot(loss_history)
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    env.close()
    
    # Return the trained policy network and history
    return policy_net, rewards_history

# Custom visualization function using matplotlib (no reliance on env.render())
def visualize_trained_agent(policy_net):
    st.header("Trained Agent Performance")
    status = st.empty()
    status.text("Running trained agent...")
    
    # Create environment for running trained agent
    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    
    # Store state history
    state_history = [state]
    reward_history = []
    
    # Run episode with trained policy
    total_reward = 0
    for t in range(500):  # Max episode length
        # Use policy with a small exploration rate
        if random.random() < 0.05:  # 5% random exploration
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        
        # Save state and reward
        state_history.append(state)
        total_reward += reward
        reward_history.append(total_reward)
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Create animation using matplotlib
    status.text(f"Creating animation... Episode completed with total reward: {total_reward}")
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    plt.tight_layout()
    
    # Function to update the animation
    def update(frame):
        # Draw cart and pole
        draw_cart_pole(state_history[frame], fig, ax1)
        
        # Update reward plot
        if frame > 0:
            ax2.clear()
            ax2.plot(reward_history[:frame], 'g-')
            ax2.set_xlim(0, len(state_history))
            ax2.set_ylim(0, total_reward + 5)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Cumulative Reward')
            ax2.set_title('Reward Over Time')
    
    # Create animation
    frames = min(100, len(state_history))  # Limit to 100 frames to avoid memory issues
    step = max(1, len(state_history) // frames)
    selected_frames = list(range(0, len(state_history), step))
    
    ani = animation.FuncAnimation(
        fig, update, frames=selected_frames, 
        interval=50, blit=False)
    
    # Save animation to buffer
    buf = io.BytesIO()
    ani.save(buf, writer='pillow', fps=15)
    buf.seek(0)
    
    # Display animation
    st.image(buf, caption=f"CartPole Agent (Total Reward: {total_reward})", width=600)
    
    # Display final stats
    st.success(f"Agent completed with reward: {total_reward}")
    
    return total_reward

# Main script
st.markdown("""
This app trains a Deep Q-Network (DQN) agent to solve the CartPole environment 
from OpenAI Gymnasium. Adjust the parameters in the sidebar to see how they 
affect the training process.

## Task Description
In CartPole, a pole is attached to a cart moving along a track. The goal is to 
balance the pole by applying forces to the cart. A reward of +1 is provided for 
every timestep that the pole remains upright. The episode ends when the pole is 
more than 15 degrees from vertical, or the cart moves more than 2.4 units from 
the center.
""")

if st.button("Train Agent"):
    with st.spinner("Training DQN agent... This might take a few minutes."):
        start_time = time.time()
        policy_net, rewards_history = train_agent()
        training_time = time.time() - start_time
        st.success(f"Training completed in {training_time:.1f} seconds!")
        
        # Plot final reward curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rewards_history)
        ax.set_title('Training Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True)
        st.pyplot(fig)
    
    # Visualize the trained agent with custom visualization
    with st.spinner("Running trained agent..."):
        try:
            reward = visualize_trained_agent(policy_net)
        except Exception as e:
            st.error(f"Error visualizing agent: {e}")
            st.info("Showing simplified performance statistics instead.")
            
            # Display text-based summary
            st.subheader("Agent Performance Summary")
            
            # Run a few test episodes
            env = gym.make("CartPole-v1")
            test_rewards = []
            
            for i in range(5):
                state, _ = env.reset()
                episode_reward = 0
                
                for t in range(500):
                    # Use policy
                    with torch.no_grad():
                        action = policy_net(torch.FloatTensor(state)).argmax().item()
                    
                    # Take action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    state = next_state
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                test_rewards.append(episode_reward)
            
            env.close()
            
            # Display results
            avg_reward = sum(test_rewards) / len(test_rewards)
            st.metric("Average Reward", f"{avg_reward:.1f}")
            st.write(f"Test Episode Rewards: {test_rewards}")
