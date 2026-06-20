from copy import deepcopy
from typing import Dict, Tuple, Optional

from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.algorithm.critic.state_value_config import StateValueConfig
from marlite.config_processor.config_processor import ConfigProcessor
from marlite.config_processor.qmix_config_processor import QMIXConfigProcessor


class QTRANConfigProcessor(ConfigProcessor):
    def parse_v_net_config(
        self, config: Dict
    ) -> Tuple[StateValueConfig, OptimizerConfig, Optional[LRSchedulerConfig]]:
        v = deepcopy(config["v_net"])
        v_opt_conf = v.pop("optimizer")
        v_opt = OptimizerConfig(**v_opt_conf)
        v_lr_conf = v.pop("lr_scheduler", None)
        v_lr = LRSchedulerConfig(**v_lr_conf) if v_lr_conf else None
        return StateValueConfig(**v), v_opt, v_lr

    def process(self, config: Dict[str, Dict]) -> Tuple[Dict, Dict, str]:
        config = deepcopy(config)
        trainer_kwargs, train_args, checkpoint = QMIXConfigProcessor().process(config)
        v_cfg, v_opt, v_lr = self.parse_v_net_config(config)
        trainer_kwargs["v_net_config"] = v_cfg
        trainer_kwargs["v_optimizer_config"] = v_opt
        trainer_kwargs["v_lr_scheduler_conf"] = v_lr
        trainer_kwargs["lambda_opt"] = config["trainer"].get("lambda_opt", 1.0)
        trainer_kwargs["lambda_nopt"] = config["trainer"].get("lambda_nopt", 1.0)
        trainer_kwargs["is_optimal_mask_mode"] = config["trainer"].get("is_optimal_mask_mode", True)
        return trainer_kwargs, train_args, checkpoint
