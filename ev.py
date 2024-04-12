import numpy as np
import math
from scipy.integrate import odeint
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import PMState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from utilities import project_state_kb, integrate_nonlin_dynamics

class EV:

    vmax = 35.0
    vmin = 0.0
    umax = np.array([5.0, 0.4])
    umin = np.array([-9.0, -0.4])
    dumax = np.array([9.0, 0.4])

    def __init__(self, initial_state, vehicle_params, controller, T):
        self._x0 = project_state_kb(initial_state) #    [s, d, phi, v].T
        self._x0_curv = False   #   curved ev state (updated externally)
        self._xx = [self._x0]
        self._initial_state = EV.generatePMstate(self._x0, 0)
        self._state_list = [self._initial_state]
        self._shape = Rectangle(length=vehicle_params.l, width=vehicle_params.w)
        self._lr = vehicle_params.l/2
        self._lflr_ratio = 0.5
        self._controller = controller
        self._T = T
    
    def run_step(self, k, x0_tv, xrefo, lane_info, comparison=False):
        # Controller update in curvilinear coordinates
        if comparison:
            u0 = self._controller.run_step_comparison(self._x0_curv, x0_tv, xrefo, lane_info, lane_info['d_max'],
                     lane_info['d_min'])
        else:
            u0 = self._controller.run_step(self._x0_curv, x0_tv, xrefo, lane_info, lane_info['d_max'],
                     lane_info['d_min'])
        #   state update takes place in cartesian coordinates
        self._x0 = integrate_nonlin_dynamics(self._x0, u0, self._T, self._lr, self._lflr_ratio)
        self._xx.append(self._x0)
        self._state_list.append(EV.generatePMstate(self._x0, k))
        self._x0_curv = False

    def get_prediction(self):
        return TrajectoryPrediction(trajectory=Trajectory(initial_time_step=0, state_list=self._state_list),
                                                            shape=self._shape)
        
    def generatePMstate(x0, k):
        return PMState(**{'position': np.array([x0[0], x0[1]]), 'time_step': k,
                            'velocity': x0[3]*math.cos(x0[2]),
                            'velocity_y': x0[3]*math.sin(x0[2])})

