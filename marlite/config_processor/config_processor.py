from typing import Dict, Tuple

class ConfigProcessor:

    def get_common_kwargs(self, config: Dict) -> Dict:
        """Process common configurations and return common kwargs for all trainers"""
        raise NotImplementedError

    def get_specific_kwargs(self, config: Dict, trainer_config: Dict) -> Dict:
        """Process the config and return specific kwargs for the trainer"""
        raise NotImplementedError

    def process(self, config: Dict) -> Tuple[Dict, Dict, str]:
        """Process the config and return trainer_kwargs, trainer_class, train_args, and checkpoint"""
        raise NotImplementedError