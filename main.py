import argparse
import yaml
from src.trainer.trainer_config import TrainerConfig

def main(config_path):
    with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    trainer_config = TrainerConfig(config)
    trainer_config.create_trainer()
    results = trainer_config.run()
    print("Training completed. Results:\n", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training process based on a configuration file.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    main(args.config)