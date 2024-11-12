# Cargar el modelo
agent.q_network.load_state_dict(torch.load('/content/drive/MyDrive/auto_model.pth'))
agent.q_network.eval()  # Establecer el modelo en modo de evaluación
