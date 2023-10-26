REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .sopcg_controller import SopcgMAC
from .dcg_controller import DeepCoordinationGraphMAC
from .scpcg_controller import ScpcgMAC
from .casec_controller import CASECMAC
from .ccg_controller import CcgMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["sopcg_mac"] = SopcgMAC
REGISTRY["scpcg_mac"] = ScpcgMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC
REGISTRY['casec_mac'] = CASECMAC
REGISTRY['ccg_mac'] = CcgMAC