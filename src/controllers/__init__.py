REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .sopcg_controller import SopcgMAC
from .dcg_controller import DeepCoordinationGraphMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["sopcg_mac"] = SopcgMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC
