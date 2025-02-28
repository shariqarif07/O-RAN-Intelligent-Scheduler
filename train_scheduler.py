
from oran_ai_scheduler import simulate_environment, save_model, load_model

if __name__ == "__main__":
    load_model()
    simulate_environment()
    save_model()
    print("Training completed!")
