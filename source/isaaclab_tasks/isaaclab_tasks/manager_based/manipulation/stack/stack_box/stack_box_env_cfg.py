import os
import isaaclab.sim as sim_utils


from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim import MeshCollisionPropertiesCfg, SDFMeshPropertiesCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObservationsCfg, RewardsCfg, StackEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.tactile_stack_env_cfg import TactileFrankaStackEnvCfg

LOCAL_BLUE_BLOCK_USD  = "./assets/Props/blue_block_sdf.usd"

@configclass
class FrankaStackBoxEnvCfg(TactileFrankaStackEnvCfg):
    """Configuration for the Franka Gelsight Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set each stacking cube deterministically
        self.scene.stack_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/stack_object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=LOCAL_BLUE_BLOCK_USD,
                scale=(1.0, 1.0, 1.0),
                rigid_props=self.cube_properties,
                semantic_tags=[("class", "stack_object")],
            ),
        )