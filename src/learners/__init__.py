from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .sopcg_learner import SopcgLearner
from .dcg_learner import DCGLearner
from .scpcg_learner import ScpcgLearner
from .casec_learner import CASECLearner
from .ccg_learner import CcgLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["sopcg_learner"] = SopcgLearner
REGISTRY["dcg_learner"] = DCGLearner
REGISTRY["scpcg_learner"] = ScpcgLearner
REGISTRY["casec_learner"] = CASECLearner
REGISTRY["ccg_learner"] = CcgLearner