import numpy as np
from commonroad.scenario.obstacle import ObstacleType
from utilities import lane_center, curv_tv

class Obstacle:
    #   legend 1st symbol: V = vehicle
    #   legend 2nd symbol: E = east, S = south, W = west, N = north
    umax = {'VE': np.array([5.0, 0.4]),
            'VS': np.array([0.4, 9.0]),
            'VW': np.array([9.0, 0.4]),
            'VN': np.array([0.4, 5.0])}
    umin = {'VE': np.array([-9.0, -0.4]),
            'VS': np.array([-0.4, -5.0]),
            'VW': np.array([-5.0, -0.4]),
            'VN': np.array([-0.4, -9.0])}
    P_w = {'VE': np.diag([0.44, 0.09]),
            'VS': np.diag([0.09, 0.44]),
            'VW': np.diag([0.44, 0.09]),
            'VN': np.diag([0.09, 0.44])}
    P_v = {'VE': np.diag([0.25, 0.25, 0.028, 0.028]),
            'VS': np.diag([0.028, 0.028, 0.25, 0.25]),
            'VW': np.diag([0.25, 0.25, 0.028, 0.028]),
            'VN': np.diag([0.028, 0.028, 0.25, 0.25])}
    Qo = {'VE': np.diag([0.0, 1.0, 0.1, 0.1]),
            'VS': np.diag([0.1, 0.1, 0.0, 1.0]),
            'VW': np.diag([0.0, 1.0, 0.1, 0.1]),
            'VN': np.diag([0.1, 0.1, 0.0, 1.0])}
    Ro = {'VE': np.diag([1.0, 0.15]),
            'VS': np.diag([0.15, 1.0]),
            'VW': np.diag([1.0, 0.15]),
            'VN': np.diag([0.15, 1.0])}
    lwo = {'VE': np.array([5.0, 2.0]),
            'VS': np.array([2.0, 5.0]),
            'VW': np.array([5.0, 2.0]),
            'VN': np.array([2.0, 5.0])}
    v_amp = {'VE': np.array([0.25, 0.25, 0.028, 0.028]),
                'VS': np.array([0.028, 0.028, 0.25, 0.25]),
                'VW': np.array([0.25, 0.25, 0.028, 0.028]),
                'VN': np.array([0.028, 0.028, 0.25, 0.25])}
    

    def __init__(self, curv_cosy0, id=0, length=0, width=0, type=None,
                    trajectory=(None, None)):
        self._id = id
        self._length = length
        self._width = width
        self._type = type
        self._trajectory_time_steps = trajectory[0]
        self._trajectory_states = trajectory[1]
        self._lsrtype = Obstacle.assign_lsrtype(0, self._trajectory_states,
                                                self._type, curv_cosy0)
        self._umin = Obstacle.umin[self._lsrtype]
        self._umax = Obstacle.umax[self._lsrtype]
        self._P_w = Obstacle.P_w[self._lsrtype]
        self._P_v = Obstacle.P_v[self._lsrtype]
        self._Qo = Obstacle.Qo[self._lsrtype]
        self._Ro = Obstacle.Ro[self._lsrtype]
        self._lw = Obstacle.lwo[self._lsrtype]
        self._v_amp = Obstacle.v_amp[self._lsrtype]

    def get_reference(self, k, lane_info, curv_cosy):
        x0_tv_curv = curv_tv(self._trajectory_states[k], curv_cosy)
        d_ref = lane_center(x0_tv_curv[2], lane_info)
        return np.array([0.0, x0_tv_curv[1], d_ref, 0.0])

    def assign_lsrtype(k, trajectory_states, type, curv_cosy0):
        if type!=ObstacleType.CAR:
            print('\n\n\nNON-CAR OBSTACLE! FIX PARAMETERS\n\n\n')
            breakpoint()
        lsr_type = 'V'
        x0_curv = curv_tv(trajectory_states[k], curv_cosy0)
        orientation = np.round(np.arctan2(x0_curv[3], x0_curv[1])/np.pi*2)    #       vy/vx
        if orientation==0:
            lsr_type += 'E'
        elif orientation==1:
            lsr_type += 'N'
        elif orientation==2:
            lsr_type += 'W'
        else: lsr_type += 'S'
        return lsr_type