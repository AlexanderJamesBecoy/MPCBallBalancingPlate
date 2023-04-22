import cvxpy as cp

"""
---------------------------------------------------------------------
Model Predictive Control. 
---------------------------------------------------------------------
This is based on L. Hoogenboom and K. Trip's MPC assignment.
To be accessed here: https://studiocodeorange.nl/MPC/MPC_assignment.pdf 
Created by: Alexander James Becoy
Date: 22-04-2023
"""

class Controller:
    """
    
    """
    
    def __init__(self, dt, N):
        """
        Initialize the parameters of the model predictive controller.
        @param: dt - discretized time step in seconds
        @param: N - prediction horizon N
        """
        self.dt = dt
        self.N = N

