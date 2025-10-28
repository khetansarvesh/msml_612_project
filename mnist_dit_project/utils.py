def log(message):
    print(f"[LOG] {message}")

def save_config(config, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(config, f)

def load_config(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)