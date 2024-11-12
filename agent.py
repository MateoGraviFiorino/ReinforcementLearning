import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Definir la red neuronal para el agente (Q-Network)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Definir el agente DQN
class Auto:
    def __init__(self, input_dim, output_dim, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=64, replay_buffer_size=10000, lr=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Inicializar la red neuronal
        self.q_network = DQN(input_dim, output_dim)
        self.target_network = DQN(input_dim, output_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
    def select_action(self, state):
        if random.random() < self.epsilon:
            # Explorar: seleccionar una acción aleatoria como un vector de 3 valores
            return np.random.uniform(low=[-1, -1, 0], high=[1, 1, 1])  # 3 valores continuos
        else:
            # Explotar: seleccionar la acción con la mayor Q-valor
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state)
            action_idx = torch.argmax(q_values).item()
            return self.get_action_from_index(action_idx)

    def get_action_from_index(self, action_idx):
        # Aquí puedes mapear el índice de la acción al vector correspondiente
        if action_idx == 0:
            return np.array([1, 0, 0])  # Acelerar
        elif action_idx == 1:
            return np.array([0, -1, 0])  # Girar a la izquierda
        elif action_idx == 2:
            return np.array([0, 1, 0])   # Girar a la derecha


    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Muestrear un minibatch de la memoria de repetición
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir listas de numpy.ndarray a numpy.array
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Calcular Q(s, a) usando la red neuronal
        q_values = self.q_network(states)  # Predicción de la red
        q_value = torch.sum(q_values * actions, dim=1)  # Producto punto con las acciones continuas

        # Calcular el valor objetivo
        next_q_values = self.target_network(next_states)
        next_q_value = next_q_values.max(1)[0]
        target = rewards + self.gamma * next_q_value * (1 - dones)

        # Calcular la pérdida (MSE)
        loss = nn.MSELoss()(q_value, target)

        # Actualizar la red neuronal
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar el valor de epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay



    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())