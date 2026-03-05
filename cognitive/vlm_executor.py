"""
cognitive/vlm_executor.py
==========================
VLM Code Executor — the robot's brain.

Sends {user_command + scene_graph} to a self-hosted Qwen3-VL-8B
and executes the Python code it returns. The VLM has direct access
to all motor APIs (loco, arm, grasp) and perception.

No fixed primitives — the VLM writes whatever code the task needs.
"""
import io
import sys
import json
import traceback
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


CODE_SYSTEM_PROMPT = '''You are a robot controller. You write Python code to control a Unitree G1 humanoid robot.

You have access to these objects (already instantiated):
- `loco` — LocomotionController: walk the robot
    loco.walk_to(x, y, tolerance=0.15) → {"success": bool, "final_distance": float}
    loco.walk_velocity(vx, vy, wz, duration_s) — send velocity for duration
    loco.stand_still(duration_s) — hold position

- `arm` — ArmController: move the right arm
    arm.move_to(target_pos_xyz, max_steps=300, tolerance=0.03) → {"success": bool, "final_error": float, "com_stable": bool}
    arm.get_ee_pos() → np.array([x, y, z])
    arm.is_reachable(target_pos) → bool
    arm.is_com_stable() → bool

- `grasp` — GraspController: pick up and release objects
    grasp.detect_nearby(hand="right", max_dist=0.15) → [{"name": str, "distance": float, "position": array}]
    grasp.grasp(hand="right", object_name=None) → {"success": bool, "object": str, "distance": float}
    grasp.release(hand="right") → {"success": bool}
    grasp.is_grasping(hand="right") → bool

- `perception` — ActivePerceptionSim: see the world
    perception.update_from_cameras() — refresh scene graph from cameras
    perception.get_scene_graph() → SceneGraph
    perception.query_object(label) → SceneObject or None (has .position_world, .label, .mass_kg)
    perception.get_graspable_objects() → [SceneObject, ...]
    perception.print_state() — show all tracked objects

- `bridge` — UnitreeBridge: raw robot state
    bridge.get_base_pos() → np.array([x, y, z])
    bridge.get_ee_pos(hand="right") → np.array([x, y, z])
    bridge.get_imu("torso") → {"angular_velocity": array, "linear_acceleration": array}

- `print(...)` — use to report status and results

RULES:
1. Write ONLY Python code. No explanations, no markdown.
2. Use the exact API above. Do not invent functions.
3. Always check results — if an action fails, try to recover or report why.
4. Keep the robot stable — if arm.is_com_stable() returns False, back off.
5. After actions that change the world, call perception.update_from_cameras() to refresh.
6. Print a summary of what happened at the end.
'''


@dataclass
class ExecutionResult:
    """Result of running VLM-generated code."""
    success: bool
    code: str
    output: str
    error: Optional[str] = None
    attempts: int = 1


class VLMExecutor:
    """
    Sends commands to the VLM, gets Python code back, executes it.
    
    Usage:
        executor = VLMExecutor(api_base, model, motor_context)
        result = executor.execute("pick up the red mug")
    """

    MAX_RETRIES = 2

    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model_name: str = "Qwen/Qwen3-VL-8B",
                 motor_context: Dict[str, Any] = None,
                 local_vlm=None):
        """
        Args:
            api_base: vLLM OpenAI-compatible API URL (used if local_vlm is None)
            model_name: model to use for code generation
            motor_context: dict of {name: object} to expose to generated code
            local_vlm: optional LocalVLM instance (skips API, runs on GPU directly)
        """
        self.api_base = api_base
        self.model_name = model_name
        self.motor_context = motor_context or {}
        self.local_vlm = local_vlm

        mode = "LOCAL" if local_vlm else f"API ({api_base})"
        print(f"[VLMExec] Initialised ({mode})")
        print(f"[VLMExec] Motor context: {list(self.motor_context.keys())}")

    def generate_code(self, command: str, scene_json: str,
                      error_feedback: str = None) -> str:
        """
        Ask the VLM to generate Python code for a command.
        Uses local_vlm if available, otherwise falls back to API.
        """
        user_msg = f"COMMAND: {command}\n\nSCENE GRAPH:\n{scene_json}"
        if error_feedback:
            user_msg += f"\n\nPREVIOUS ATTEMPT FAILED:\n{error_feedback}\nFix the code."

        full_prompt = CODE_SYSTEM_PROMPT + "\n\n" + user_msg

        if self.local_vlm:
            # Local mode — direct GPU inference
            code = self.local_vlm.chat(full_prompt, max_tokens=1500, temperature=0.2)
        else:
            # API mode — vLLM server
            from openai import OpenAI
            client = OpenAI(base_url=self.api_base, api_key="dummy")
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": CODE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            code = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if code.startswith("```"):
            lines = code.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code

    def execute_code(self, code: str) -> tuple:
        """
        Execute generated Python code with motor context.
        
        Returns:
            (success: bool, output: str, error: str or None)
        """
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            # Build exec globals with motor context + standard libs
            exec_globals = {
                "__builtins__": __builtins__,
                "np": np,
                "print": print,
                "range": range,
                "len": len,
                "abs": abs,
                "min": min,
                "max": max,
                "list": list,
                "dict": dict,
                "str": str,
                "float": float,
                "int": int,
                "bool": bool,
                "True": True,
                "False": False,
                "None": None,
            }
            exec_globals.update(self.motor_context)

            exec(code, exec_globals)

            output = captured.getvalue()
            return True, output, None

        except Exception as e:
            output = captured.getvalue()
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return False, output, error

        finally:
            sys.stdout = old_stdout

    def execute(self, command: str, scene_json: str = None) -> ExecutionResult:
        """
        Full pipeline: command → VLM generates code → execute → retry on failure.
        
        Args:
            command: natural language command
            scene_json: scene graph JSON (if None, uses empty)
        """
        if scene_json is None:
            scene_json = "{}"

        print(f"\n[VLMExec] Command: '{command}'")

        error_feedback = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"[VLMExec] Attempt {attempt}/{self.MAX_RETRIES}...")

            # Generate code
            try:
                code = self.generate_code(command, scene_json, error_feedback)
            except Exception as e:
                print(f"[VLMExec] VLM API error: {e}")
                return ExecutionResult(
                    success=False, code="", output="",
                    error=f"VLM API error: {e}", attempts=attempt
                )

            print(f"[VLMExec] Generated code ({len(code)} chars):")
            print("--- CODE ---")
            print(code)
            print("--- END ---")

            # Execute code
            success, output, error = self.execute_code(code)

            if success:
                print(f"[VLMExec] ✅ Success!")
                if output:
                    print(f"[Output] {output}")
                return ExecutionResult(
                    success=True, code=code, output=output,
                    error=None, attempts=attempt
                )
            else:
                print(f"[VLMExec] ❌ Failed: {error}")
                error_feedback = f"Code:\n{code}\n\nError:\n{error}"

        # All retries failed
        return ExecutionResult(
            success=False, code=code, output=output,
            error=error, attempts=self.MAX_RETRIES
        )
