import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Soft Actor Network
class SoftActor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(SoftActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    # With action scaling
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        actions = torch.tanh(x_t)
        
        # Modify actions based on their index
        scaled_actions = torch.zeros_like(actions)
        for i in range(actions.size(-1)):
            if i % 4 == 0:
                scaled_actions[:, i] = actions[:, i]
            else:
                scaled_actions[:, i] = (actions[:, i] + 1) / 2
        
        log_prob = normal.log_prob(x_t) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return scaled_actions, log_prob


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1).to(device)  
        return self.q(sa)  # Returns a single Q-value tensor


# Temperature
class Temperature(nn.Module):
    def __init__(self, initial_value=1.0):
        super(Temperature, self).__init__()
        self.log_alpha = nn.Parameter(torch.tensor([initial_value], dtype=torch.float32))
        self.target_entropy = -torch.tensor(1.0)  # Placeholder for target entropy, adjust as needed

    def forward(self):
        return self.log_alpha.exp()

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, replay_buffer):
        self.actor = SoftActor(state_dim, action_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)
        self.temperature = Temperature(initial_value=1.5).to(device)
        self.replay_buffer = replay_buffer
        self.highest_total_reward = -np.inf  # Placeholder for highest total reward

        # Initialize target critic networks to match critic networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=3e-4)
        self.temperature_optimizer = Adam([self.temperature.log_alpha], lr=3e-4)
        
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            actions, _ = self.actor.sample(state)
        return actions.cpu().numpy()[0]
    
    def update_parameters(self, states, actions, rewards, next_states, dones, discount, tau):
        # Update Critic Networks
        with torch.no_grad():
            next_actions, log_probs = self.actor.sample(next_states)
            # Now each critic returns a single tensor directly
            target_Q1 = self.target_critic_1(next_states, next_actions)
            target_Q2 = self.target_critic_2(next_states, next_actions)
            target_V = torch.min(target_Q1, target_Q2) - self.temperature() * log_probs
            target_Q = rewards + (1 - dones) * discount * target_V
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Update Actor Network
        new_actions, log_probs = self.actor.sample(states)
        Q1_new = self.critic_1(states, new_actions)
        Q2_new = self.critic_2(states, new_actions)
        actor_loss = (self.temperature() * log_probs - torch.min(Q1_new, Q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        with torch.no_grad():
            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update Temperature
        alpha_loss = -(self.temperature.log_alpha * (log_probs + self.temperature.target_entropy).detach()).mean()

        self.temperature_optimizer.zero_grad()
        alpha_loss.backward()
        self.temperature_optimizer.step()
    

    def save_models(self, save_dir):
        """
        Saves the state dictionaries of the actor, critics, and temperature models.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the actor model
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))

        # Save the critic models
        torch.save(self.critic_1.state_dict(), os.path.join(save_dir, 'critic_1.pth'))
        torch.save(self.critic_2.state_dict(), os.path.join(save_dir, 'critic_2.pth'))

        # Save the target critic models (optional, if you want to fully restore training)
        torch.save(self.target_critic_1.state_dict(), os.path.join(save_dir, 'target_critic_1.pth'))
        torch.save(self.target_critic_2.state_dict(), os.path.join(save_dir, 'target_critic_2.pth'))

        # Save the temperature model (if using a learnable temperature)
        torch.save(self.temperature.state_dict(), os.path.join(save_dir, 'temperature.pth'))

        print("Models saved to:", save_dir)

    def load_models(self, load_dir):
        """
        Loads the state dictionaries of the actor, critics, and temperature models from the specified directory.
        """
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, 'actor.pth'), map_location=device))
        self.critic_1.load_state_dict(torch.load(os.path.join(load_dir, 'critic_1.pth'), map_location=device))
        self.critic_2.load_state_dict(torch.load(os.path.join(load_dir, 'critic_2.pth'), map_location=device))
        self.target_critic_1.load_state_dict(torch.load(os.path.join(load_dir, 'target_critic_1.pth'), map_location=device))
        self.target_critic_2.load_state_dict(torch.load(os.path.join(load_dir, 'target_critic_2.pth'), map_location=device))
        self.temperature.load_state_dict(torch.load(os.path.join(load_dir, 'temperature.pth'), map_location=device))

        print("Models loaded from:", load_dir)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
