from agent import Auto
import gymnasium as gym
from tqdm import tqdm

# Inicializar el entorno y el agente
env = gym.make('CarRacing-v3')
input_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]  # Tamaño de la imagen (estados)
output_dim = 3  # 3 acciones posibles: izquierda, derecha, acelerar
agent = Auto(input_dim=input_dim, output_dim=output_dim)

# Entrenamiento del agente
num_episodios = 100
for episodio in range(num_episodios):
    state, info = env.reset()
    state = state.flatten()  # Aplanar la imagen de entrada
    done = False
    total_recompensa = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = next_state.flatten()  # Aplanar la imagen de entrada
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_recompensa += reward

    print(f'Episodio {episodio + 1}/{num_episodios} - Recompensa total: {total_recompensa}')
    agent.update_target_network()

# Cerrar el entorno después del entrenamiento
env.close()