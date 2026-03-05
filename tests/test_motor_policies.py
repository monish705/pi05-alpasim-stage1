"""Quick test of RL locomotion + arm IK + grasp."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
from motor.unitree_bridge import UnitreeBridge
from motor.locomotion import LocomotionController
from motor.arm_controller import ArmController
from motor.grasp_controller import GraspController

SCENE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mujoco_menagerie", "unitree_g1", "g1_tabletop.xml"
)

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
bridge = UnitreeBridge(model, data)
bridge.reset_to_stand()

# 1. Locomotion
print("\n=== TEST 1: RL Locomotion ===")
loco = LocomotionController(bridge)

print("\nStanding with RL policy (2s)...")
loco.stand_still(duration_s=2.0)
base = bridge.get_base_pos()
print(f"After stand: z={base[2]:.3f}m (stable: {base[2] > 0.5})")

print("\nWalking to (0.3, 0.0)...")
result = loco.walk_to(0.3, 0.0, max_duration_s=8.0)
print(f"Success: {result['success']}, dist: {result['final_distance']:.3f}m")
base = bridge.get_base_pos()
print(f"Final: ({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f})")

# 2. Arm IK
print("\n=== TEST 2: CoM-Aware Arm IK ===")
arm = ArmController(bridge, hand="right")
ee_pos = arm.get_ee_pos()
print(f"EE start: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")

target = base + [0.3, -0.15, 0.1]  # relative to current base
print(f"Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
ik_result = arm.move_to(target, max_steps=300)
print(f"IK: success={ik_result['success']}, error={ik_result['final_error']:.4f}m, "
      f"com_stable={ik_result['com_stable']}, abort={ik_result['abort_reason']}")

# 3. Grasp
print("\n=== TEST 3: Grasp ===")
grasp = GraspController(bridge)
nearby = grasp.detect_nearby("right", max_dist=0.5)
print(f"Nearby objects: {len(nearby)}")
for obj in nearby:
    print(f"  {obj['name']}: {obj['distance']:.3f}m")

print("\n=== DONE ===")
