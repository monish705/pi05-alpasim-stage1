# 🦾 Unitree G1 Embodied AI: VLM-RL-IK Control Pipeline

![Status](https://img.shields.io/badge/Status-Prototype-orange)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Sim](https://img.shields.io/badge/MuJoCo-3.1.2-red)
![License](https://img.shields.io/badge/License-MIT-green)

A complete, end-to-end embodied AI architecture for the Unitree G1 humanoid robot. This project demonstrates a state-of-the-art hierarchical control pipeline: translating high-level natural language semantic goals into low-level joint torques in a physics-accurate MuJoCo simulation.

**Why this project?**
Most open-source robotics projects stop at the VLM (Vision-Language Model) or the RL (Reinforcement Learning) locomotion policy. This project bridges the gap, employing highly sophisticated CoM-aware Jacobian Inverse Kinematics (IK) to allow a walking humanoid to perform autonomous arm manipulation based on VLM-generated code.

## 🧠 Architecture Stack

The system is built on a 5-layer hierarchy:

1.  **Semantic Brain (VLM Executor):** Uses a Vision-Language Model (like Qwen3-VL-8B or pi0) to process RGB camera input and user text prompts, generating executable Python action code.
2.  **Cognitive Middleware:** Translates the VLM's Python code into coordinate-space goals.
3.  **Whole-Body Manipulation (ArmController):** A custom analytical IK solver that uses damped least squares to calculate joint angles for reaching/grasping, *while explicitly checking Support Polygon and Center of Mass (CoM) margins to prevent the robot from tipping over*.
4.  **Balance & Locomotion (RL Policy):** An ONNX-exported neural network policy running at 50Hz that keeps the G1 balanced and walking toward targets.
5.  **Motor & Physics (MuJoCo):** The low-level `UnitreeBridge` that translates IK positional targets and RL velocity commands into raw actuator control (`data.ctrl`) in the MuJoCo physics engine.

## 🚀 Getting Started

### Prerequisites

*   Python 3.11+
*   MuJoCo 3.1.2+
*   `mujoco_menagerie` (Unitree G1 assets) cloned into your home directory:
    ```bash
    git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
    ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unitree-embodied-ai.git
   cd unitree_embodied_ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

To launch the autonomous pick-and-place simulation:

```bash
python main.py
```

## 🎥 Demo Capabilities
*   **VLM Code Generation:** Live translation of "pick up the red mug" into `motor.arm.move_to(mug_pos)`
*   **Autonomous Grasping:** Weld-constraint simulated grasping of objects on a table.
*   **Whole-Body Balancing:** The robot dynamically shifts its waist and legs to counter the weight shift of its arms reaching forward.

## 📄 License
MIT License
