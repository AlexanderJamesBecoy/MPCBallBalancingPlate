import cvxpy as cp
import numpy as np

"""
---------------------------------------------------------------------
Model Predictive Control. 
---------------------------------------------------------------------
This is based on L. Hoogenboom and K. Trip's MPC assignment.
To be accessed here: https://studiocodeorange.nl/MPC/MPC_assignment.pdf 
Created by: Alexander James Becoy
Date: 22-04-2023
"""

class MPC:
    """
    
    """
    
    def __init__(self, sys, dt, N):
        """
        Initialize the parameters of the model predictive controller.
        @param: sys - system to-be controlled, ball balancing plate in this case.
        @param: dt - discretized time step in seconds
        @param: N - prediction horizon N
        """
        self.sys = sys
        self.dt = dt
        self.N = N
        # self.Q = np.eye(sys._NO_OF_STATES) # TODO
        # self.R = np.eye(sys._NO_OF_INPUTS) # TODO

        self.Q = np.diag([1.,0.,1.,0.,1.,0.,1.,0.])
        self.R = 0.2 * np.eye(sys._NO_OF_INPUTS)
        self.P = self.calculate_DARE()

    def predict(self, x_init, x_target):
        """
        Compute
        @param: x_init - initial state
        @param: x_target - target state
        @return: MPC input, next state, plan for visualization
        """
        cost = 0.0
        constraints = []

        # Create optimization variables
        x = cp.Variable((self.sys._NO_OF_STATES, self.N + 1))
        u = cp.Variable((self.sys._NO_OF_INPUTS, self.N))

        # For each stage in k = 0, ..., N-1
        constraints += [x[:, 0] == x_init]

        for k in range(self.N):
            # Compute the cost at time instant k
            cost += self.calculate_cost(x[:, k+1], u[:, k], x[:,self.N])
            
            # Consider constraints at time instant k
            constraints += [
                # System dynamics
                x[:, k+1] == self.sys.A @ x[:, k] + self.sys.B @ u[:, k],
                # x[:, 0] == x_init,

                # State constraints
                x[:, k] >= -1.0 * self.sys.constraints['system'],
                x[:, k] <= self.sys.constraints['system'],

                # Input constraints
                u[:, k] >= -1.0 * self.sys.constraints['input'],
                u[:, k] <= self.sys.constraints['input'],

                # TODO: x(N) in terminal set ?????
            ]

        # Solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP) # verbose=True
        # print(f"Status: {problem.status}")

        # Assert that a solution is found
        assert u[:, 0].value is not None and x[:, 1].value is not None, "No feasible solution is found."

        return u[:, 0].value, x[:, 1].value, x[:, :].value, None

    def calculate_cost(self, x, u, x_N):
        """
        Calculate the cost with the stage cost and the terminal cost.
        @param: x - state at some time instant k+1.
        @param: u - input at some time instant k.
        @param: x_N - state at furthest foreseeable time instant N.
        @return: cost - resulting cost to minimize for optimal control.
        """
        # Calculate the stage function.
        cost = cp.quad_form(x, self.Q) + cp.quad_form(u, self.R)

        # # Obtain P using Discrete Algebraic Riccati Equation (DARE).
        # P = self.Q
        # for n in range(self.N):
        #     P1 = (self.sys.A.T @ P) @ self.sys.A
        #     P2 = (self.sys.A.T @ P) @ self.sys.B
        #     P3 = np.linalg.inv(self.R + (self.sys.B.T @ P) @ self.sys.B)
        #     P4 = (self.sys.B.T @ P) @ self.sys.A
        #     P =  P1 - (P2 @ P3) @ P4 + self.Q
        
        # Calculate the terminal cost with the found matrix P.
        cost += cp.quad_form(x_N, self.P)

        return cost
    
    def calculate_DARE(self):
        """
        Obtain matrix P to calculate the terminal cost using
        Discrete Algebraic Riccati Equation (DARE).
        @return P: _NO_OF_STATES x _NO_OF_STATES matrix
        """
        # Initialize P as Q
        P = self.Q

        # Calculate P using DARE for N iterations
        for n in range(self.N):
            P1 = (self.sys.A.T @ P) @ self.sys.A
            P2 = (self.sys.A.T @ P) @ self.sys.B
            P3 = np.linalg.inv(self.R + (self.sys.B.T @ P) @ self.sys.B)
            P4 = (self.sys.B.T @ P) @ self.sys.A
            P =  P1 - (P2 @ P3) @ P4 + self.Q

        return P
    
    def generate_terminal_set(self):
        """
        Generate a terminal set to ensure stability and feasibility.
        
        @return T: positively invariant list of points? TODO
        """
