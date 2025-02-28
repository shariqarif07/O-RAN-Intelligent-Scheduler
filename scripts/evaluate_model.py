
import torch
from oran_ai_scheduler import model, simulate_environment

def evaluate_model():
    model.load_state_dict(torch.load("models/dqn_scheduler.pth"))
    print("Model loaded successfully. Running evaluation...")
    simulate_environment(episodes=100)
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate_model()
