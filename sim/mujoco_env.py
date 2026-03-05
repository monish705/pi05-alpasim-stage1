import mujoco
import mujoco.viewer
import numpy as np
import cv2

class MuJoCoEnv:
    def __init__(self, xml_path, render_mode="window"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.width = 640
        self.height = 480
        
        # Initialize headless renderer for sensor simulation
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.enable_depth_rendering()
        
        self.viewer = None
        self.render_mode = render_mode
        if self.render_mode == "window":
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.sync()
            except Exception as e:
                print(f"[SIM] Window mode failed ({e}), falling back to headless")
                self.render_mode = "headless"

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def get_rgb(self, camera_name):
        self.renderer.update_scene(self.data, camera=camera_name)
        self.renderer.disable_depth_rendering()
        return self.renderer.render()

    def get_depth(self, camera_name):
        self.renderer.update_scene(self.data, camera=camera_name)
        # MuJoCo returns depth in meters natively if enabled on context,
        # but the Python binding renderer typically scales it. 
        # By default renderer.render() outputs depth as 0-1 mapped values.
        # Let's extract the raw depth maps from the OpenGL context
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        
        # MuJoCo provides depth buffer mapped differently depending on near/far clip
        # Real distance translation needs focal logic if we want exact meters
        # For phase 1, returning the raw float depth map from renderer
        return depth

    def get_camera_intrinsics(self, camera_name):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self.model.cam_fovy[cam_id]
        
        # Calculate focal length
        f = 0.5 * self.height / np.tan(fovy * np.pi / 360)
        
        return {
            "fx": f,
            "fy": f,
            "cx": self.width / 2,
            "cy": self.height / 2
        }

if __name__ == "__main__":
    import time
    from pathlib import Path
    
    scene_path = Path(__file__).parent / "scenes" / "tabletop.xml"
    if not scene_path.exists():
        print(f"Error: {scene_path} not found")
        exit(1)
        
    print(f"Loading scene: {scene_path}")
    env = MuJoCoEnv(str(scene_path), render_mode="window")
    env.reset()
    
    print("Simulating 500 steps to let objects settle...")
    for _ in range(500):
        env.step()
        time.sleep(0.005)
        
    print("\nFetching Sensor Data:")
    
    # 1. RGB
    rgb = env.get_rgb("overview_cam")
    print(f"RGB Camera 'overview_cam': {rgb.shape}, dtype: {rgb.dtype}")
    
    # Save a frame locally to verify
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_frame.png", bgr)
    print("Saved 'test_frame.png'")
    
    # 2. Depth
    depth = env.get_depth("overview_cam")
    print(f"Depth Camera 'overview_cam': {depth.shape}, dtype: {depth.dtype}")
    print(f"Depth range: min={depth.min():.4f}, max={depth.max():.4f}")
    
    # 3. Intrinsics
    K = env.get_camera_intrinsics("overview_cam")
    print(f"Camera Intrinsics: {K}")
    
    print("\nClosing viewer in 3 seconds...")
    time.sleep(3)
    if env.viewer:
        env.viewer.close()
