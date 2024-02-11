import os
import numpy as np
import time
from agent import SACAgent, ReplayBuffer
from game import SoccerEnv
import torch
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    # Initialize the environment
    env = SoccerEnv()

    # Initialization parameters
    batch_size = 256
    discount = 0.99
    tau = 0.005  # Target network update rate
    learning_rate = 3e-4
    alpha_lr = 3e-4  # Learning rate for temperature

    state_dim = 36
    action_dim = 16  # Assuming continuous action space

    # Directory for saving models
    weights_dir = "./models"
    os.makedirs(weights_dir, exist_ok=True)

    # Replay Buffer
    replay_buffer = ReplayBuffer(capacity=1000000)

    # SAC Agent Initialization
    sac_agent = SACAgent(state_dim, action_dim, replay_buffer)

    # Load models if available
    if os.path.exists(weights_dir):
        sac_agent.load_models(weights_dir)
    else:
        print("Load directory does not exist. Starting training from scratch.")

    def update_sac():
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        # Directly convert numpy arrays to tensors on the specified device
        states = torch.tensor(states, device=device, dtype=torch.float)
        actions = torch.tensor(actions, device=device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float)
        dones = torch.tensor(dones, device=device, dtype=torch.float).unsqueeze(1)

        sac_agent.update_parameters(states, actions, rewards, next_states, dones, discount, tau)


    num_episodes = 10000  # Number of episodes to train
    update_every = 50
    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state, dtype=np.float32)  # Ensure state is a float32 numpy array
        episode_reward = 0
        done = False
        steps = 0  # Track steps within the episode
        while not done:
            action = sac_agent.select_action(np.array(state))
            # Check if the ball is inside the table before taking action
            if env.ball_check():
                next_state, reward, done = env.step(action.tolist())
                next_state = np.array(next_state, dtype=np.float32)  # Ensure next_state is a float32 numpy array
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                steps += 1
                # Check if it's time to update the SAC agent
                if steps % update_every == 0 and replay_buffer.__len__() >= batch_size:
                    update_sac()

            time.sleep(0.1)  # Delay for simulating real-time actions

        print(f"Episode {episode}: Total Reward: {episode_reward}")

        # Save models
        if episode_reward > sac_agent.highest_total_reward:
            sac_agent.highest_total_reward = episode_reward
            sac_agent.save_models(weights_dir)
            print(f"New best model saved with reward: {sac_agent.highest_total_reward}")
