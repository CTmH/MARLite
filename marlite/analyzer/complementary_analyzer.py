"""Experimental: Capture-first evaluation using terminal episode metrics, not lagged infos."""

from marlite.analyzer.analyzer import Analyzer


class ComplementaryAnalyzer(Analyzer):
    def __call__(self, episodes):
        if any("capture_fraction" not in episode for episode in episodes):
            raise ValueError(
                "Complementary analyzer requires the complementary rollout profile"
            )
        results = {
            "capture_fraction": self._calculate_statistics(
                [e["capture_fraction"] for e in episodes]
            )
        }
        results["communication_bytes"] = self._calculate_statistics(
            [
                sum(sum(step.values()) for step in e["communication_bytes"])
                for e in episodes
            ]
        )
        for key in ("effective_hit_rate", "missed_attack_rate"):
            results[key] = self._calculate_statistics([e[key] for e in episodes])
        results.update(super().__call__(episodes))
        return results
