
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the DQN model for scheduling
class DQNScheduler(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNScheduler, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
INPUT_DIM = 10  # Features: CQI, SINR, PRB usage, Traffic load, etc.
OUTPUT_DIM = 5  # Actions: PRB allocation, MCS selection, Power control
LR = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 32

# Instantiate model and optimizer
model = DQNScheduler(INPUT_DIM, OUTPUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Define experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer()

# Training function
def train():
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)
    
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = model(next_states).max(1)[0].detach()
    target_q_values = rewards + (GAMMA * next_q_values * ~dones)
    
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Simulation environment
def simulate_environment(episodes=1000):
    global EPSILON
    for episode in range(episodes):
        state = np.random.rand(INPUT_DIM)  # Placeholder for real RAN state
        done = False
        while not done:
            if random.random() < EPSILON:
                action = random.randint(0, OUTPUT_DIM - 1)
            else:
                with torch.no_grad():
                    action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            
            next_state = np.random.rand(INPUT_DIM)
            reward = random.uniform(-1, 1)
            done = random.choice([True, False])
            
            replay_buffer.push(state, action, reward, next_state, done)
            train()
            state = next_state
        
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

# Save and Load Model
import os

def save_model():
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dqn_scheduler.pth")
    print("Model saved successfully.")

def load_model():
    if os.path.exists("models/dqn_scheduler.pth"):
        model.load_state_dict(torch.load("models/dqn_scheduler.pth"))
        print("Model loaded successfully.")
    else:
        print("No pre-trained model found. Training from scratch.")
