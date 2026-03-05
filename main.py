"""
main.py
========
Main orchestrator — ties everything together.

Usage:
    # Interactive mode (requires self-hosted Qwen3-VL-8B via vLLM):
    python main.py

    # Single command:
    python main.py --command "pick up the red mug"

    # Offline test (no VLM, just perception + motors):
    python main.py --offline
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
from motor.unitree_bridge import UnitreeBridge
from motor.locomotion import LocomotionController
from motor.arm_controller import ArmController
from motor.grasp_controller import GraspController
from perception.active_sim import ActivePerception
from cognitive.vlm_executor import VLMExecutor
from sparse_sim.sandbox import SparseSandbox


SCENE_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mujoco_menagerie", "unitree_g1", "g1_tabletop.xml"
)


class EmbodiedAISystem:
    """
    Full Embodied AI Pipeline:
      Perception (SAM3 + Qwen3-VL) → VLM Code Generation → Motor Execution
    """

    def __init__(self, scene_xml: str = SCENE_XML,
                 vlm_api_base: str = "http://localhost:8000/v1",
                 vlm_model: str = "Qwen/Qwen3-VL-8B",
                 local_vlm=None):
        print("=" * 60)
        print("  Unitree G1 Embodied AI System")
        print("=" * 60)

        # 1. Simulation
        print("\n[1/5] Loading MuJoCo scene...")
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        # 2. Motor layer
        print("[2/5] Initialising motor controllers...")
        self.bridge = UnitreeBridge(self.model, self.data)
        self.bridge.reset_to_stand()
        self.loco = LocomotionController(self.bridge)
        self.arm = ArmController(self.bridge, hand="right")
        self.grasp = GraspController(self.bridge)

        # 3. Active perception (same code works for sim + real robot)
        print("[3/5] Starting active perception...")
        self.perception = ActivePerception.from_sim(
            self.model, self.data,
            bridge=self.bridge,
            grasp=self.grasp,
            vlm_api_base=vlm_api_base,
            vlm_model=vlm_model,
        )

        # 4. Sparse sim sandbox
        print("[4/5] Initialising sparse sim sandbox...")
        self.sandbox = SparseSandbox(scene_xml)

        # 5. VLM code executor (the brain)
        print("[5/5] Connecting VLM executor...")
        motor_context = {
            "loco": self.loco,
            "arm": self.arm,
            "grasp": self.grasp,
            "perception": self.perception,
            "bridge": self.bridge,
            "sandbox": self.sandbox,
            "np": np,
        }
        self.executor = VLMExecutor(
            api_base=vlm_api_base,
            model_name=vlm_model,
            motor_context=motor_context,
            local_vlm=local_vlm,
        )

        # Store local VLM ref for perception tagging
        self.local_vlm = local_vlm

        print("\n" + "=" * 60)
        print("  System Ready!")
        print("=" * 60)

    def perceive(self):
        """Run one perception cycle."""
        return self.perception.update_from_cameras(["head_cam"])

    def execute_command(self, command: str) -> dict:
        """
        Execute a natural language command.
        
        1. Perceive the scene
        2. Give scene graph + command to VLM
        3. Execute VLM's code
        4. Re-perceive to verify
        """
        print(f"\n{'='*60}")
        print(f"  COMMAND: {command}")
        print(f"{'='*60}")

        # Step 1: Perceive
        print("\n[Step 1] Perceiving scene...")
        scene_graph = self.perceive()
        scene_json = self.perception.get_scene_graph_json()
        self.perception.print_state()

        # Step 2 + 3: VLM generates code → execute
        print("\n[Step 2] Generating and executing code...")
        result = self.executor.execute(command, scene_json)

        # Step 4: Re-perceive to verify
        print("\n[Step 3] Verifying result...")
        self.perceive()
        self.perception.print_state()

        # Report
        bridge_pos = self.bridge.get_base_pos()
        ee_pos = self.bridge.get_ee_pos("right")
        print(f"\n[Result] success={result.success}, "
              f"attempts={result.attempts}")
        print(f"[Robot]  base=({bridge_pos[0]:.2f}, {bridge_pos[1]:.2f}, "
              f"{bridge_pos[2]:.2f})")
        print(f"[Robot]  hand=({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, "
              f"{ee_pos[2]:.2f})")
        if self.grasp.is_grasping():
            print(f"[Robot]  holding: {self.grasp.get_grasped_object()}")

        return {
            "success": result.success,
            "code": result.code,
            "output": result.output,
            "error": result.error,
        }

    def run_interactive(self):
        """Interactive command loop."""
        print("\nType commands or 'quit' to exit.")
        print("Examples:")
        print("  > pick up the red mug")
        print("  > walk to the table")
        print("  > what objects do you see?")
        print()

        while True:
            try:
                command = input("🤖 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not command:
                continue
            if command.lower() in ("quit", "exit", "q"):
                break

            self.execute_command(command)

    def run_offline_demo(self):
        """
        Demo without VLM — just motor policies + perception.
        Useful for testing locally without a GPU server.
        """
        print("\n[Offline Demo] Running without VLM...")

        # Stand
        print("\n[1] Standing...")
        self.loco.stand_still(1.0)
        print(f"  Base: {self.bridge.get_base_pos()}")

        # Walk toward table
        print("\n[2] Walking to table...")
        result = self.loco.walk_to(0.3, 0.0, max_duration_s=5.0)
        print(f"  Walk: {result['success']}, dist={result['final_distance']:.3f}")

        # Detect objects
        print("\n[3] Detecting nearby objects...")
        nearby = self.grasp.detect_nearby("right", max_dist=0.5)
        for obj in nearby:
            print(f"  {obj['name']}: {obj['distance']:.3f}m")

        # Reach
        if nearby:
            target = nearby[0]
            print(f"\n[4] Reaching for {target['name']}...")
            reach_target = target['position'] + np.array([0, 0, 0.03])
            result = self.arm.move_to(reach_target, max_steps=200)
            print(f"  IK: success={result['success']}, err={result['final_error']:.4f}")

            if result['success']:
                print("\n[5] Grasping...")
                grasp_result = self.grasp.grasp("right", target['name'])
                print(f"  Grasp: {grasp_result}")

        print("\n[Offline Demo] Complete!")


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 Embodied AI")
    parser.add_argument("--command", type=str, help="Single command to execute")
    parser.add_argument("--offline", action="store_true",
                        help="Run offline demo (no VLM)")
    parser.add_argument("--vlm-url", type=str,
                        default="http://localhost:8000/v1",
                        help="vLLM API URL")
    parser.add_argument("--vlm-model", type=str,
                        default="Qwen/Qwen3-VL-8B",
                        help="VLM model name")
    args = parser.parse_args()

    system = EmbodiedAISystem(
        vlm_api_base=args.vlm_url,
        vlm_model=args.vlm_model,
    )

    if args.offline:
        system.run_offline_demo()
    elif args.command:
        system.execute_command(args.command)
    else:
        system.run_interactive()


if __name__ == "__main__":
    main()
