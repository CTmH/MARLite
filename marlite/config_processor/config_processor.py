from typing import Dict, Tuple

class ConfigProcessor:

    def process(self, config: Dict) -> Tuple[Dict, Dict, str]:
        """Process the config and return trainer_kwargs, trainer_class, train_args, and checkpoint"""
        raise NotImplementedError