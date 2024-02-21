import mujoco
import numpy as np
from viewer import Viewer
from time import sleep
import os

xml_path = os.path.join(os.path.dirname(__file__), "assets", "model.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
viewer = Viewer(model, data)

if __name__ == "__main__":

    print("Press Ctrl+C to stop the simulation.")

    print([model.joint(i).name for i in range(model.njnt)])

    while True:

        # Make joint1_z move up and down
        data.ctrl[0] = 2.0 * np.sin( 2 * np.pi * data.time)
        data.ctrl[1] = 2.0 * np.abs(np.sin( 2 * np.pi * data.time + np.pi/2))
        data.ctrl[2] = 2.0 * np.abs(np.sin( 2 * np.pi * data.time + np.pi))
        data.ctrl[3] = 2.0 * np.abs(np.sin( 2 * np.pi * data.time + 3*np.pi/2))

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        viewer.update()