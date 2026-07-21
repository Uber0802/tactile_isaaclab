from .tactile_reward_model import TactileRewardModel

# Deliberately no logging.NullHandler() here. The classic library advice is to
# attach one, but that predates Python 3.2's logging.lastResort: a NullHandler
# satisfies the handler search, so lastResort never fires and WARNING-level
# messages (failed ckpt import, curve-dump failures) go silent in scripts that
# never configure logging. Without it we get the behavior we want for free —
# warnings reach stderr with zero setup, INFO stays quiet until a host asks.

__all__ = ["TactileRewardModel"]
