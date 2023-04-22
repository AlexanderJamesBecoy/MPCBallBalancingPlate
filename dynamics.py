import numpy as np # TODO: Replace to JAX?

"""
---------------------------------------------------------------------
System dynamics of a ball balancing plate. 
---------------------------------------------------------------------
This is based on L. Hoogenboom and K. Trip's MPC assignment.
To be accessed here: https://studiocodeorange.nl/MPC/MPC_assignment.pdf 
Created by: Alexander James Becoy
Date: 21-04-2023

Notations:
- COR: center of rotation

Controlled inputs = {Fx_1, Fx_2, Fy_1, Fy_2}
Measured = {yb}
Unknowns = {gamma, dd_gamma, Nb}
"""

""""
TODO: Update function on each as function of alpha, gamma, N, etc.
 - save current dynamics in self
 - get self
"""

class LinearModel:
    """
    Ball balancing plate linearized around the origin.
    """
    _NO_OF_STATES = 8
    _NO_OF_INPUTS = 4
    _NO_OF_OUTPUTS = 2
    
    def __init__(self, dt, m, g, l, vmax, wmax, Tmax, rmax, I=None, linear=True):
        """
        Initialize the linearized ball balancing plate and store the parameters.
        @param: dt - time instance
        @param: m - mass of the ball.
        @param: g - gravitational acceleration.
        @param: l - length of the plate.
        @param: I - moment of inertia of the plate.
        """
        # Store the parameters.
        self.dt = dt
        self.m = m
        self.g = g
        self.l = l
        if I is None:
            self.I = (4.0 * m * l**2) / 3.0 # (m * l**2) / 6.0
        else:
            self.I = I

        # Define the state matrix A.
        dd_v_factor = 0.6 * g
        dd_ang_factor = -m * g / self.I
        A = np.array([
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., dd_v_factor, 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., dd_v_factor, 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [dd_ang_factor, 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., dd_ang_factor, 0., 0., 0., 0., 0.],
        ]) 

        # Define the input matrix B.
        dd_F_factor = -0.5 * l / self.I
        if linear:
            B = np.array([
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [dd_F_factor, 0., dd_F_factor, 0.],
                [0., 0., 0., 0.],
                [0., dd_F_factor, 0., dd_F_factor],
            ])
        else:
            B = np.array([
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., -l/(2*self.I), 0., -l/(2*self.I), 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., -l/(2*self.I), 0., -l/(2*self.I)],
            ])

        # Define the output matrix C.
        C = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
        ])

        # Euler discretization
        self.A = np.ones_like(A) + A * dt
        self.B = B * dt
        self.C = C

        # Initialize constraints
        self.set_constraints(vmax, wmax, Tmax, rmax)

    def set_constraints(self, vmax, wmax, Tmax, rmax):
        """
        Define and store the system and input constraints as Dict given by the paper.
        Some system constraints are defined arbitrarily -> we make it generalizable.
        @param: vmax = maximum velocity vector (d_x, d_y) in x and y-directions.
        @param: wmax = maximum angular velocity vector (d_pitch, d_roll) in pitch and roll.
        @param: Tmax = maximum torque that can be exerted by a motor.
        @param: rmax = length of the maximum moment arm.
        """
        # Calculate constraints
        dmax = self.l / 2.0
        ang = np.pi / 4.0
        Fmax = Tmax / rmax
        # sys_constraint = np.array([dmax, vmax[0], dmax, vmax[1], ang, wmax[0], ang, wmax[1]])
        # input_constraint = Fmax * np.ones(self._NO_OF_INPUTS)

        constraints = {
            'system': np.array([dmax, vmax[0], dmax, vmax[1], ang, wmax[0], ang, wmax[1]]),
                # 'x': self.l / 2.0, # m
                # 'd_x': vmax[0], # m/s
                # 'y': self.l / 2.0, # m
                # 'd_y': vmax[1], # m/s
                # 'pitch': np.pi/4, # rad
                # 'd_pitch': wmax[0], # rad/s
                # 'roll': np.pi/4, # rad
                # 'd_roll': wmax[1], # rad/s
            #     'min': -1.0 * sys_constraint,
            #     'max': sys_constraint,
            # },
            'input': Fmax * np.ones(self._NO_OF_INPUTS),
                # Tmax / rmax, # N
            #     'min': -1.0 * input_constraint,
            #     'max': input_constraint,
            # }
        }
        self.constraints = constraints

    def update(self, x, u):
        """
        Calculate the next state of the ball balancing state given the current state and inputs.
        @param: x - current state of the system
        @param: u - force inputs
        @return: next state of the system
        """
        # Assert that state vector x has valid length.
        assert len(x) == self._NO_OF_STATES, f"Parameter x should have length {self._NO_OF_STATES}, input has length {len(x)}."
        # Assert that input vector u has valid length.
        assert len(u) == self._NO_OF_INPUTS, f"Parameter u should have length {self._NO_OF_INPUTS}, input has length {len(u)}."

        # return np.dot(self.A, x) + np.dot(self.B, u)
        return self.A @ x + self.B @ u
    
    def get_output(self, x):
        """
        Obtain the position of the ball given the current state.
        @param: x - current state of the system
        @return: current position of the ball
        """
        # Assert that state vector x has valid length.
        assert len(x) == self._NO_OF_STATES, f"Parameter x should have length {self._NO_OF_STATES}, input has length {len(x)}."

        # return np.dot(self.C, x)
        return self.C @ x

class Plate:
    """
    Dynamics of the plate model.
    """
    def __init__(self, m, l, I=None):
        """
        Initialize by storing the values for every parameter of the plate model.
        @param: m - the mass of the plate.
        @param: l - the length of the plate.
        @param: I - the inertia of the plate.
        """
        self.m = m
        self.l = l
        if I is not None:
            self.I = I
        else:
            # Calculate inertia based on square
            self.I = (m * l**2) / 6.0

    def get_R_world(self, alpha, gamma):
        """
        Obtain the frame transformation from the world frame F_w to the plate frame F_p.
        """
        p_F_w = np.array([
            [np.cos(gamma), -np.sin(gamma) * np.cos(alpha), np.sin(gamma) * np.sin(alpha)],
            [np.sin(gamma), np.cos(gamma) * np.cos(alpha), -np.cos(gamma) * np.sin(alpha)],
            [0.0, np.sin(alpha), np.cos(alpha)],
        ])
        return p_F_w

    def get_dd_y(self, Fx, gamma, N):
        """
        Obtain the acceleration in the y-axis of the plate model as function of two motors.
        Rule: Delta_theta_1 = -Delta_theta_2 -> dd_y = 0
        @param: Fx - force vector of two motors in the x-axis.
        @param: gamma - roll angle at COR of the plate w.r.t. the world frame, around the x-axis.
        @param: N - normal force on the plate.
        """
        # Calculate force difference between the two motors.
        F = np.dot(np.array([[-1.0, 0.0],[0.0, 1.0]]), Fx)

        # Calculate the acceleration in the y-axis.
        dd_y = (F - np.sin(gamma) * N) / self.m
        return dd_y
    
    def get_dd_z(self, Fy, gamma, N):
        """
        Obtain the acceleration in the y-axis of the plate model as function of two motors.
        Rule: Delta_theta_1 = -Delta_theta_2 -> dd_z = 0
        @param: Fy - force vector of two motors in the y-axis.
        @param: gamma - roll angle at COR of the plate w.r.t. the world frame, around the x-axis.
        @param: N - normal force on the plate.
        """
        # Calculate force difference between the two motors.
        F = np.dot(np.array([[1.0, 0.0],[0.0, 1.0]]), Fy)

        # Calculate the acceleration in the y-axis.
        dd_z = (F - np.cos(gamma) * N) / self.m - self.g
        return dd_z
    
    def get_dd_gamma(self, Fx, Fy, gamma, N, d):
        """
        Obtain the angular acceleration of the roll angle w.r.t. the world, around the x-axis.
        @param: Fx - force vector of two motors in the x-axis.
        @param: Fy - force vector of two motors in the y-axis.
        @param: gamma - roll angle at COR of the plate w.r.t. the world frame, around the x-axis.
        @param: N - normal force on the plate.
        @param: d - distance between the ball and COR; (0 at COR, < 0 in -y-direction, > 0 in +y-direction)
        """
        # Calculate the moment in x and y-axes.
        Mx = np.sin(gamma) * (self.l / 2.0) * np.dot(np.array([[1.0, 0.0],[0.0, 1.0]]), Fx)
        My = np.cos(gamma) * (self.l / 2.0) * np.dot(np.array([[-1.0, 0.0],[0.0, 1.0]]), Fy)

        # Calculate the angular acceleration of the roll angle.
        dd_gamma = (Mx + My - d * N) / self.I
        return dd_gamma

class Ball:
    """
    Dynamics of the ball model.
    """
    def __init__(self, m, r, mu, g):
        """
        Initialize by storing the values for every parameter of the ball model.
        @param: m - the mass of the ball.
        @param: r - the radius of the ball.
        @param: mu - the dry Coulomb friction constant.
        @param: g - gravitational acceleration.
        """
        self.m = m
        self.r = r
        self.mu = mu
        self.g = g

    def get_W_constraint(self, gamma):
        """
        Obtain the constraint friction force on the ball assuming no slip occurs between the ball and the plate.
        @param: gamma - roll angle at COR of the plate w.r.t. the world frame, around the x-axis.
        """
        W = 2.0 * np.sin(gamma) * self.m * self.g / 5.0
        return W

    def get_ddy(self, gamma, N):
        """
        Obtain the acceleration in the y-axis of the ball model.
        @param: gamma - roll angle at COR of the plate w.r.t. the world frame, around the x-axis.
        @param: N - normal force on the ball.
        """
        # Obtain the friction force on the ball.
        W = self.mu * N

        # Calculate for the acceleration in the y-axis.
        y = (np.sin(gamma) * N - np.cos(gamma) * W) / self.m
        return y
    
    def get_ddz(self, gamma, N):
        """
        Obtain the acceleration in the z-axis of the ball model.
        @param: gamma - roll angle at COR of the plate w.r.t. the world frame, around the x-axis.
        @param: N - normal force on the ball.
        """
        # Obtain the friction force on the ball.
        W = self.mu * N

        # Calculate for the acceleration in the z-axis.
        z = (np.cos(gamma) * N + np.sin(gamma) * W) / self.m - self.g

    def get_w_yb(self, gamma):
        """
        Obtain the angular velocity of the ball w.r.t. the plate assuming no slip between the ball and the plate.
        @param: gamma - roll angle at COR of the plate w.r.t. the world frame, around the x-axis.
        """
        # Obtain the specific friction force on the ball.
        W = self.get_W(gamma)

        # Calculate for the angular velocity of the ball w.r.t. the plate.
        w_yb = 3.0 * W / (2.0 * self.m * self.r)
        return w_yb