# Plots for paper
# Imports
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.visualization.draw_params import DynamicObstacleParams
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %% Define Plot tasks
Task = []
Task.append(["20-10-2023_09:36:48_USA_US101-13_2_T-1"])
Task.append(["20-10-2023_09:35:33_DEU_A99-1_2_T-1"])


# %% loop throu all tasks
for task in Task:
    # %% Load data
    path_data = "data/" + task[0]
    data = pickle.load(open(path_data, 'rb'))
    # upack dict
    for (k, v) in data.items():
      exec(f'{k} = data[k]')

    # %% Creating plots
    #   get EV trajectory and create commonroad object (using DynamicObstacle)
    ego_vehicle = DynamicObstacle(obstacle_id=id_ev, obstacle_type=ObstacleType.CAR,
                                  obstacle_shape=ev._shape, initial_state=ev._initial_state,
                                  prediction=ev.get_prediction())
    ego_params = DynamicObstacleParams()
    ego_params.vehicle_shape.occupancy.shape.facecolor = "r"

    text = controller.applied_method.copy()
    text.insert(0, 'Initial state')
    #   plot scenario and EV for each time step
    for k in range(len(ev._xx)):
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        rnd.draw_params.time_begin = k
        scenario.draw(rnd)
        ego_params.time_begin = k
        ego_vehicle.draw(rnd, draw_params=ego_params)
        # planning_problem_set.draw(rnd)
        rnd.render()
        # draw reference path
        # plt.plot(ref_path[:, 0], ref_path[:, 1], zorder=100, linewidth=2, color='green')
        plt.title("Iteration " + str(k) + ": " + text[k])
        plt.xlim([np.array(ev._xx)[:,0].min()-50, np.array(ev._xx)[:,0].max()+50])
        plt.ylim([np.array(ev._xx)[:,2].min()-50, np.array(ev._xx)[:,2].max()+20])
        plt.show(block=False)
        plt.close()

    # %% Creating plots  tim
    #   get EV trajectory and create commonroad object (using DynamicObstacle)
    ego_vehicle_comparison = DynamicObstacle(obstacle_id=id_ev, obstacle_type=ObstacleType.CAR,
                                      obstacle_shape=ev_comparison._shape, initial_state=ev_comparison._initial_state,
                                      prediction=ev_comparison.get_prediction())

    ego_params_comparison = DynamicObstacleParams()
    ego_params_comparison.vehicle_shape.occupancy.shape.facecolor = "r"
    text = controller.applied_method_comparison.copy()
    text.insert(0, 'Initial state')
    #   plot scenario and EV for each time step
    for k in range(len(ev_comparison._xx)):
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        rnd.draw_params.time_begin = k
        scenario.draw(rnd)
        ego_params_comparison.time_begin = k
        ego_vehicle_comparison.draw(rnd, draw_params=ego_params_comparison)
        # planning_problem_set.draw(rnd)
        rnd.render()
        # draw reference path
        # plt.plot(ref_path[:, 0], ref_path[:, 1], zorder=100, linewidth=2, color='green')
        plt.title("Iteration " + str(k) + ": " + text[k])
        plt.xlim([np.array(ev_comparison._xx)[:,0].min()-50, np.array(ev_comparison._xx)[:,0].max()+50])
        plt.ylim([np.array(ev_comparison._xx)[:,2].min()-50, np.array(ev_comparison._xx)[:,2].max()+20])
        plt.show(block=False)
        plt.close()