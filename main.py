import os
import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from vehiclemodels import parameters_vehicle3
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.visualization.draw_params import DynamicObstacleParams
from commonroad.scenario.state import InitialState
from utilities import compute_lqr, extract_trajectory, get_frenet_frame, curv_ev, curv_tv, get_lane_info
from obstacle import Obstacle
from controller import SMPCVPM
from ev import EV
import copy
import pickle
from datetime import datetime
np.set_printoptions(suppress=True)

# %% Startup
#  Set parameters and flags
N = 10
enable_comparison = True
iter_recompute_cosy = False 
show_plots = True 
number_runs_code = 1
# number_runs_code = 100    # Number of Simulations to evaluate statistical properties of the computation time

# %% Scenarios for the simulation
# Select the file name of the scenario
file_name = 'USA_US101-13_2_T-1' 
# file_name = 'DEU_A99-1_2_T-1'

# -----------------------------------------------------------------------------
# %% Load scenario
features_store = ['__cost', '__cost_comparison',
                  '__time_smpc_average', '__time_smpc_max',
                  '__time_cvpm1_average', '__time_cvpm1_max',
                  '__time_cvpm2_average', '__time_cvpm2_max',
                  '__time_isfeasible_average', '__time_isfeasible_max',
                  '__time_cvpmcase_average', '__time_cvpmcase_max',
                  '__time_smpcbranch_average', '__time_smpcbranch_max',
                  '__time_cvpmbranch_average', '__time_cvpmbranch_max',
                  '__time_smpc_comparison_average', '__time_smpc_comparison_max',
                  '__time_ftpnow_comparison_average', '__time_ftpnow_comparison_max',
                  '__time_ftpnext_comparison_average', '__time_ftpnext_comparison_max',
                  '__time_smpcbranch_comparison_average', '__time_smpcbranch_comparison_max',
                  '__time_ftpbranch_comparison_average', '__time_ftpbranch_comparison_max',
                  ]
store_variables = {key: [] for key in features_store}
for idx in range(number_runs_code):  # Loop to run  a number of Simulations to evaluate statistical properties of the computation time
    #   read scenario html
    file_path = os.path.join(os.getcwd(), 'scenario/'+file_name+'.xml')
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    n_steps = scenario._dynamic_obstacles[[*scenario._dynamic_obstacles.keys()][0]]._prediction._final_time_step
    T = scenario._dt

    if len(planning_problem_set._planning_problem_dict.keys()) != 1:
        breakpoint()
    id_ev = [*planning_problem_set._planning_problem_dict.keys()][0]
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # Extract obstacle lists
    Ao = np.array([[1.0, T, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, T], [0.0, 0.0, 0.0, 1.0]])
    Bo = np.array([[0.5*T**2, 0.0], [T, 0.0], [0.0, 0.5*T**2], [0.0, T]])
    obstacle_list = []
    Ko, umaxo, umino, P_w, P_v, lwo, v_amp = [], [], [], [], [], [], []
    curv_cosy0, ref_path0, dir_ref_path0 = get_frenet_frame(scenario, planning_problem)  # initial frenet frame
    for o in scenario._dynamic_obstacles.values():
        obstacle_list.append(Obstacle(curv_cosy0, id=o._obstacle_id, length=o._obstacle_shape._length, width=o._obstacle_shape._width,
                                    type=o._obstacle_type, trajectory=extract_trajectory(o)))
        Ko.append(compute_lqr(Ao, Bo, obstacle_list[-1]._Qo, obstacle_list[-1]._Ro))
        umaxo.append(obstacle_list[-1]._umax)
        umino.append(obstacle_list[-1]._umin)
        P_w.append(obstacle_list[-1]._P_w)
        P_v.append(obstacle_list[-1]._P_v)
        lwo.append(obstacle_list[-1]._lw)
        v_amp.append(obstacle_list[-1]._v_amp)

    print('run=',idx)
    #   generate EV object
    controller = SMPCVPM(T, N, Ao, Bo, Ko, umaxo, umino, P_w, P_v, lwo, v_amp, Obstacle.lwo['VE'],
                        parameters_vehicle3.parameters_vehicle3(), EV.vmax, EV.vmin, EV.umax,
                        EV.umin, EV.dumax, xref_ev=np.array([0.0, 0.0, 0.0, 27.0]),
                        Q_ev=np.diag([0.0, 2.5, 10.0, 0.25]), R_ev=np.diag([0.33, 15]), beta=0.9,
                        features_store=features_store)
    if [*planning_problem_set._planning_problem_dict.keys()] != [id_ev]:
        print('Error: Planning problem set does not contain ', id_ev, ' only')

    ev = EV(planning_problem_set.find_planning_problem_by_id(id_ev)._initial_state,
            parameters_vehicle3.parameters_vehicle3(), controller, T)


    #   double variables for comparison algorithm
    planning_problem_set_comparison = copy.deepcopy(planning_problem_set)
    scenario_comparison = copy.deepcopy(scenario)
    planning_problem_comparison = list(planning_problem_set_comparison.planning_problem_dict.values())[0]
    ev_comparison = EV(planning_problem_set_comparison.find_planning_problem_by_id(id_ev)._initial_state,
            parameters_vehicle3.parameters_vehicle3(), controller, T)

    # simulation loop ev
    curv_cosy = curv_cosy0
    ref_path = ref_path0
    dir_ref_path = dir_ref_path0
    curv_cosy_comparison = curv_cosy0
    ref_path_comparison = ref_path0
    dir_ref_path_comparison = dir_ref_path0
    for k in range(1, n_steps+1):
        if iter_recompute_cosy:
            curv_cosy, ref_path, dir_ref_path = get_frenet_frame(scenario, planning_problem)
        ev._x0_curv = curv_ev(ev._x0, curv_cosy, scenario=scenario, ref_path = ref_path,
                            planning_problem_set=planning_problem_set)
        lane_info = get_lane_info(scenario, curv_cosy, ev._x0_curv,
                                ref_path=ref_path,
                                planning_problem_set=planning_problem_set,
                                dir_ref_path = dir_ref_path)
        x0_tv = []
        xrefo = []
        if iter_recompute_cosy:
            curv_cosy_comparison, ref_path_comparison, dir_ref_path_comparison = get_frenet_frame(scenario_comparison, planning_problem_comparison)
        ev_comparison._x0_curv = curv_ev(ev_comparison._x0, curv_cosy_comparison,  scenario=scenario, ref_path = ref_path_comparison,
                            planning_problem_set=planning_problem_set)
        lane_info_comparison = get_lane_info(scenario_comparison, curv_cosy_comparison, ev_comparison._x0_curv,
                                ref_path=ref_path_comparison,
                                planning_problem_set=planning_problem_set_comparison,
                                dir_ref_path = dir_ref_path_comparison)
        x0_tv_comparison = []
        xrefo_comparison = []
        
        for o in obstacle_list:
            x0_tv.append(curv_tv(o._trajectory_states[k], curv_cosy))
            xrefo.append(o.get_reference(k, lane_info, curv_cosy))
            x0_tv_comparison.append(curv_tv(o._trajectory_states[k], curv_cosy_comparison))
            xrefo_comparison.append(o.get_reference(k, lane_info_comparison, curv_cosy_comparison))
        ev.run_step(k, x0_tv, xrefo, lane_info, comparison=False)  # controller uses projected state tvs
        planning_problem._initial_state = InitialState(time_step=k+1, position=ev._x0[:2],
                                                    orientation=ev._x0[2], velocity=ev._x0[3])
        if enable_comparison:
            ev_comparison.run_step(k, x0_tv_comparison, xrefo_comparison, lane_info_comparison, comparison=True)  # controller uses projected state tvs
            planning_problem_comparison._initial_state = InitialState(time_step=k+1, position=ev_comparison._x0[:2],
                                                    orientation=ev_comparison._x0[2], velocity=ev_comparison._x0[3])

    controller.print_computation_times()
    controller.print_average_stage_cost()
    for key in store_variables.keys():
        exec('store_variables[\'%s\'].append(controller._SMPCVPM%s)'%(key, key))
    

# %% Creating plots
if show_plots:
    #   get EV trajectory and create commonroad object (using DynamicObstacle)
    ego_vehicle = DynamicObstacle(obstacle_id=id_ev, obstacle_type=ObstacleType.CAR,
                                obstacle_shape=ev._shape, initial_state=ev._initial_state,
                                prediction=ev.get_prediction())
    ego_params = DynamicObstacleParams()
    ego_params.vehicle_shape.occupancy.shape.facecolor = "r"
    # scenario._dynamic_obstacles[id_ev] = ego_vehicle


    #   plot scenario and EV for each time step
    for k in range(n_steps):
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        rnd.draw_params.time_begin = k
        scenario.draw(rnd)
        ego_params.time_begin = k
        ego_vehicle.draw(rnd, draw_params=ego_params)
        # planning_problem_set.draw(rnd)
        rnd.render()
        plt.title("Iteration " + str(k+1) + ": " + ev._controller.applied_method[k])
        plt.xlim([np.array(ev._xx)[:,0].min()-50, np.array(ev._xx)[:,0].max()+50])
        plt.ylim([np.array(ev._xx)[:,2].min()-50, np.array(ev._xx)[:,2].max()+20])
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
        
        
    # %% Creating plots for comparison
    #   get EV trajectory and create commonroad object (using DynamicObstacle)
    ego_vehicle_comparison = DynamicObstacle(obstacle_id=id_ev, obstacle_type=ObstacleType.CAR,
                                obstacle_shape=ev_comparison._shape, initial_state=ev_comparison._initial_state,
                                prediction=ev_comparison.get_prediction())

    ego_params_comparison = DynamicObstacleParams()
    ego_params_comparison.vehicle_shape.occupancy.shape.facecolor = "r"
    #   plot scenario and EV for each time step
    for k in range(n_steps):
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        rnd.draw_params.time_begin = k
        scenario.draw(rnd)
        ego_params_comparison.time_begin = k
        ego_vehicle_comparison.draw(rnd, draw_params=ego_params_comparison)
        # planning_problem_set.draw(rnd)
        rnd.render()
        plt.title("Iteration " + str(k+1) + ": " + ev_comparison._controller.applied_method_comparison[k])
        plt.xlim([np.array(ev_comparison._xx)[:,0].min()-50, np.array(ev_comparison._xx)[:,0].max()+50])
        plt.ylim([np.array(ev_comparison._xx)[:,2].min()-50, np.array(ev_comparison._xx)[:,2].max()+20])
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
    
    
# %% Print scenations  
controller.print_applied_method(comparison=False)
controller.print_applied_method(comparison=True)

# %% Save necessary Data for plots of the paper 
now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
data = {
        "ev" : ev,
        "id_ev": id_ev,
        "controller": controller,
        "scenario" : scenario,
        }
    
if enable_comparison:
    data_comparison = {
            "ev_comparison" : ev_comparison,
            }
    data.update(data_comparison)


if not os.path.isdir('data'):
    os.mkdir('data')
pickle.dump(data, open('data/'+now + '_' + file_name , 'wb'))
pickle.dump(store_variables, open('data/'+file_name+'_'+now + '_' + 'variables_many_runs' , 'wb'))
