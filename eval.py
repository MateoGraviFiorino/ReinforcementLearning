import torch
import numpy as np
import gymnasium as gym
from agent import Auto

# Configuración del entorno
env = gym.make('CarRacing-v3', render_mode='human')  # Renderizado en tiempo real

# Configuración de parámetros (deben coincidir con los del entrenamiento)
input_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
output_dim = 3

# Inicializar el agente
agent = Auto(input_dim, output_dim)

# Cargar el modelo entrenado
model_path = 'auto_model.pth'  # Ruta del modelo guardado
agent.q_network.load_state_dict(torch.load(model_path))
agent.q_network.eval()  # Modo de evaluación

# Evaluación del modelo
num_episodes = 10  # Número de episodios para evaluar
for episode in range(num_episodes):
    state, info = env.reset()
    state = np.array(state, dtype=np.float32).flatten()  # Aplanar el estado inicial
    total_reward = 0
    done = False

    while not done:
        # Seleccionar acción usando el modelo entrenado
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Agregar batch dimension
        with torch.no_grad():  # No calcular gradientes
            q_values = agent.q_network(state_tensor)
            action = agent.get_action_from_index(torch.argmax(q_values).item())

        # Tomar acción en el entorno
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Actualizar el estado y acumular la recompensa
        state = np.array(next_state, dtype=np.float32).flatten()
        total_reward += reward

    print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f}")

# Cerrar el entorno después de evaluar
env.close()
