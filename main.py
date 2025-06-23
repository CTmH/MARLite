import argparse
import yaml
from src.trainer.trainer_config import TrainerConfig
from src.analyzer.analyzer_config import AnalyzerConfig

def train(config_path):
    with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    trainer_config = TrainerConfig(config)
    trainer_config.create_trainer()
    results = trainer_config.run()
    print("Training completed. Results:\n", results)

def analyze(config_path, output_path):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Create configuration objects
    analyzer_config = AnalyzerConfig(config_data)
    analyzer = analyzer_config.create_analyzer()
    analysis_results = analyzer.comprehensive_analysis()

    # Save results to YAML file
    with open(output_path, 'w') as f:
        yaml.safe_dump(analysis_results, f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training or analysis process based on a configuration file.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model using a configuration file')
    train_parser.add_argument('--config', type=str, required=True, help='Path to the YAML training configuration file')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze the model and output results to a YAML file')
    analyze_parser.add_argument('--config', type=str, required=True, help='Path to the YAML analysis configuration file')
    analyze_parser.add_argument('--output', type=str, required=True, help='Path to the output YAML file')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.config)
    elif args.command == 'analyze':
        analyze(args.config, args.output)