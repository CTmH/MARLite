from torch import Tensor
from torch import bernoulli, ones, argmax, stack

from .agent_group import AgentGroup
from ..model import RNNModel

class QMIXAgentGroup(AgentGroup):
    def __init__(self, 
                agents, 
                model_configs,
                feature_extractors,
                optim,
                lr: float = 1e-3,
                device: str = 'cpu') -> None:
        super().__init__(agents, model_configs, feature_extractors, optim, lr, device)

    def get_q_values(self, observations, eval_mode=True):
        """
        Get the Q-values for the given observations.

        Args:
            observations (list of Tensor): List of observation tensors.
            eval_mode (bool): Whether to set models to evaluation mode.

        Returns:
            Tensor: Concatenated Q-value tensor across all agents.
        """

        if eval_mode:
            for model in self.models.values():
                model.eval()
        else:
            for model in self.models.values():
                model.train()

        q_values = [None for _ in range(len(self.agents))]

        for (model_name, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            #idx = [i for i, agent in enumerate(self.agents) if agent[1] == model_name]
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            obs = [Tensor(observations[ag]) for ag in selected_agents]
            selected_hidden_states = [self.hidden_states[ag] for ag in selected_agents]
            obs = stack(obs).to(device=self.device)
            
            if isinstance(model, RNNModel):
                selected_hidden_states = stack(selected_hidden_states).to(device=self.device)  # N, (D * \text{num\_layers}, H_{out})
                selected_hidden_states = selected_hidden_states.permute(1, 0, 2) # (D * \text{num\_layers}, N, H_{out})
                bs = obs.shape[0]
                ts = obs.shape[1]
                obs_shape = obs.shape[2:]
                obs = obs.reshape(bs*ts, *obs_shape)
                feature = fe(obs)
                feature = feature.reshape(bs, ts, -1)
                qv, hs = model(feature, selected_hidden_states)
                # qv shape: torch.Size([2, 1, 5]) (N：batch size, L: seq length, D * H_{out})
                # hs shape: torch.Size([1, 2, 128]) (D * \text{num\_layers}, N, H_{out})
                qv = qv[:,-1,:] # get the last output (N, D * H_{out})
                hs = hs.permute(1, 0, 2) # (N, D * \text{num\_layers}, H_{out})
            else:
                # TODO: Add code for handling other types of models (e.g., CNNs)
                qv = model(obs)
                hs = [None for _ in range(len(idx))]

            for i, ag, q, h in zip(idx, selected_agents, qv, hs):
                self.hidden_states[ag] = h
                q_values[i] = q

        q_values = stack(q_values).to(device=self.device)

        return q_values

    def act(self, observations, avail_actions, epsilon, eval_mode=True):
        """
        Select actions based on Q-values and exploration.

        Args:
            observations (list of Tensor): List of observation tensors.
            avail_actions (dict): Dictionary mapping agent IDs to available action distributions.
            epsilon (float): Exploration rate.
            eval_mode (bool): Whether to set models to evaluation mode.

        Returns:
            numpy array: Selected actions for each agent.
        """
        self.init_hidden_states()
        q_values = self.get_q_values(observations, eval_mode)
        q_values = q_values.detach().to('cpu')
        random_choices = bernoulli(epsilon * ones(len(self.agents)))

        random_actions = [avail_actions[key].sample() for key in avail_actions.keys()]
        random_actions = Tensor(random_actions).to(device='cpu')

        actions = random_choices * random_actions \
            + (1 - random_choices) * argmax(q_values, axis=-1)
        actions = actions.detach().to('cpu').numpy()
        actions = actions.astype(int).tolist()

        actions = {agent_id: action for agent_id, action in zip(self.agents.keys(), actions)}
        
        return actions
