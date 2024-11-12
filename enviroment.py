import gymnasium as gym
import numpy as np
import torch
from agent import Auto  # Asumiendo que tu clase `Auto` está en agent.py

# Configuración del entorno
env = gym.make('CarRacing-v3', render_mode='human')

# Configuración de parámetros
input_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]  # Redimensionar la imagen 3D en un vector 1D
output_dim = 3  # Las 3 acciones continuas: aceleración, dirección, frenado

# Inicialización del agente
agent = Auto(input_dim, output_dim)

# Parámetros de entrenamiento
num_episodes = 1000
max_timesteps = 1000

# Ciclo de entrenamiento
for episode in range(num_episodes):
    state, info = env.reset()  # Reinicia el entorno
    state = np.array(state, dtype=np.float32)  # Aseguramos que el estado es un array de tipo float32
    total_reward = 0
    done = False
    
    for t in range(max_timesteps):
        # El agente selecciona una acción
        action = agent.select_action(state)

        # Tomar la acción en el entorno
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Almacenar la experiencia en la memoria de repetición
        agent.store_experience(state, action, reward, next_state, done)
        
        # Entrenar el agente
        agent.train()

        # Actualizar el estado
        state = np.array(next_state, dtype=np.float32)
        
        # Acumular la recompensa total
        total_reward += reward

        if done:
            break

    # Imprimir el progreso
    print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f}")

    # Actualizar la red objetivo cada cierto número de episodios
    if (episode + 1) % 10 == 0:
        agent.update_target_network()

    # Reducir la epsilon después de cada episodio
    if agent.epsilon > agent.min_epsilon:
        agent.epsilon *= agent.epsilon_decay

# Cerrar el entorno después de entrenar
env.close()
