import mujoco

class Viewer:
    """
    A simple viewer to visualize the simulation. The code is based on the following implementation: 
    https://gitlab.com/project-march/march/-/blob/main/ros2/src/mujoco_simulation/src/mujoco_sim/mujoco_sim/mujoco_visualize.py?ref_type=heads.
    """

    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data
        
        self.cam = mujoco.MjvCamera()
        self.cam.type = 1
        self.option = mujoco.MjvOption()

        mujoco.glfw.glfw.init()
        self.window = mujoco.glfw.glfw.create_window(800, 800, "MPC Balancing Plate", None, None)
        mujoco.glfw.glfw.make_context_current(self.window)
        mujoco.glfw.glfw.swap_interval(1)

        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.option)
        self.cam.trackbodyid = 0
        self.cam.distance = 0.5
        self.cam.elevation = -20.0
        self.cam.azimuth = 135.0
        # self.cam.lookat[2] += 0.1
        
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.simstart = data.time

    def update(self):
        viewport = mujoco.MjrRect(0, 0, 800, 800)

        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)

        mujoco.glfw.glfw.swap_buffers(self.window)
        mujoco.glfw.glfw.poll_events()