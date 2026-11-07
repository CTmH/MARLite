"""Experimental CP04 collection accounting and an optional exact training-step cap."""

from marlite.util.randomness import derive_seed


class AuditedRollouts:
    """Track successful native episodes without changing the default Trainer path."""

    record_cp04_diagnostics = True

    def __init__(self, config, step_budget=None):
        self.config = config
        self.step_budget = step_budget
        self.collection_steps = 0
        self.diagnostic_steps = 0
        self.batches = []
        self.training_diagnostics = []
        self.seeds = set()

    @property
    def collection_exhausted(self):
        return self.step_budget is not None and self.collection_steps == self.step_budget

    def record_diagnostics(self, epoch, metrics):
        self.training_diagnostics.append({"epoch": epoch, **metrics})

    def create_manager(self, *args, **kwargs):
        return _AuditedManager(self.config.create_manager(*args, **kwargs), self, False)

    def create_eval_manager(self, *args, **kwargs):
        return _AuditedManager(self.config.create_eval_manager(*args, **kwargs), self, True)


class _AuditedManager:
    def __init__(self, manager, audit, diagnostic):
        self.manager = manager
        self.audit = audit
        self.diagnostic = diagnostic

    def generate_episodes(self):
        manager, audit = self.manager, self.audit
        if not self.diagnostic and audit.step_budget is not None:
            remaining = audit.step_budget - audit.collection_steps
            if remaining <= 0:
                raise RuntimeError("CP04 training-step budget already exhausted")
            manager.n_episodes = min(manager.n_episodes, remaining)
            # shortcut: cap this batch uniformly, refine per-episode limits if needed.
            manager.episode_limit = min(manager.episode_limit, remaining // manager.n_episodes)
        seeds = [derive_seed(manager.seed, 0, index) for index in range(manager.n_episodes)]
        if None in seeds or audit.seeds.intersection(seeds):
            raise RuntimeError("CP04 observed an unseeded or duplicate rollout")
        episodes = manager.generate_episodes()
        lengths = [int(episode["episode_length"]) for episode in episodes]
        if len(lengths) != len(seeds) or not all(0 < n <= manager.episode_limit for n in lengths):
            raise RuntimeError("CP04 received an incomplete rollout batch")
        audit.seeds.update(seeds)
        if self.diagnostic:
            audit.diagnostic_steps += sum(lengths)
        else:
            audit.collection_steps += sum(lengths)
        audit.batches.append({
            "stream": "trainer_diagnostic" if self.diagnostic else "collection",
            "episode_limit": manager.episode_limit,
            "seeds": seeds,
            "steps": lengths,
        })
        return episodes
