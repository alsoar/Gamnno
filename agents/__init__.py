"""Named complete-turn agents for magnno. The historical replay policy lives in heuristic.py."""

from .policies import (
    AgentConfig, AssassinAgent, BaselineAgent, BlockaderAgent, FortressAgent, HoarderAgent,
    OptimizerAgent, RandomAgent, SaboteurAgent, SprinterAgent,
)


REGISTRY = {cls.name: cls for cls in (BaselineAgent, SprinterAgent, FortressAgent, SaboteurAgent,
            AssassinAgent, BlockaderAgent, OptimizerAgent, HoarderAgent, RandomAgent)}
AGENT_NAMES = tuple(REGISTRY)


def create_agent(name, config=None, seed=None):
    """Construct one policy from its stable lowercase id and optional AgentConfig or mapping."""
    if name not in REGISTRY:
        raise ValueError(f"unknown agent {name!r}; choose from {', '.join(AGENT_NAMES)}")
    return REGISTRY[name](config=config, seed=seed)
