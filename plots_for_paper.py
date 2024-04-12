# Plots for paper
# Imports
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.visualization.draw_params import DynamicObstacleParams
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# %%  flags
make_pgf = True
# make_pgf = False

# make_tikz = True
make_tikz = False

if make_tikz:
    import tikzplotlib 

add_captions = True
# add_captions = False

moresteps = True
# moresteps = False

# %% Define Plot tasks
Task = []
# Task.append(["Dataset", [xmin, xmax, ymin, ymax], [ list of frames ], [figsize]])
Task.append(["20-10-2023_16:32:15_USA_US101-13_2_T-1", [-10, 50, -50, 10], [1,20,21,24] , [3,4]])
# Task.append(["20-10-2023_13:37:25_DEU_A99-1_2_T-1", [-30, 100, -5, 15], [1,10,23,24,25] , [3,2]])
Task.append(["03-08-2023_10:40:04_DEU_A99-1_2_T-1", [-30, 45, -5, 15], [1] , [3,2]])
Task.append(["03-08-2023_10:40:04_DEU_A99-1_2_T-1", [35, 110, -5, 15], [23] , [3,2]])

output_text = ""


if not os.path.exists('png'):
   os.makedirs('png')
if make_pgf:
    if not os.path.exists('pgf'):
       os.makedirs('pgf')
if make_tikz:
    if not os.path.exists('tikz'):
       os.makedirs('tikz')

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
    for k in task[2]:
        plt.figure()
        rnd = MPRenderer(plot_limits=task[1],figsize=(25,10))
        if moresteps:            
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1dd0ea"
            rnd.draw_params.time_begin = k+2
            scenario.draw(rnd)
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1db0ea"
            rnd.draw_params.time_begin = k+1
            scenario.draw(rnd)
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1d7eea"
            ego_params.vehicle_shape.occupancy.shape.facecolor = "#ffa0a0"
            ego_params.time_begin = k +2
            ego_vehicle.draw(rnd, draw_params=ego_params)
            ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff8080"
            ego_params.time_begin = k +1
            ego_vehicle.draw(rnd, draw_params=ego_params)
            ego_params.vehicle_shape.occupancy.shape.facecolor = "r"
            
        rnd.draw_params.time_begin = k
        scenario.draw(rnd)
        ego_params.time_begin = k
        ego_vehicle.draw(rnd, draw_params=ego_params)
        rnd.render()
        title = "Iteration " + str(k) + ": " + text[k]
        rnd.ax.set_title(title)
        output_text = output_text + title + '\n'
        filename = 'png/' + task[0] +'_frame_' +str(k) + '.png'
        rnd.f.savefig(filename, bbox_inches='tight')
        plt.close()
        if make_pgf:
            plt.figure()
            rnd = MPRenderer(plot_limits=task[1],figsize=(task[3][0], task[3][1]))
            if moresteps:            
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1dd0ea"
                rnd.draw_params.time_begin = k+2
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1db0ea"
                rnd.draw_params.time_begin = k+1
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1d7eea"
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ffa0a0"
                ego_params.time_begin = k +2
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff8080"
                ego_params.time_begin = k +1
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "r"
            rnd.draw_params.time_begin = k
            scenario.draw(rnd)
            ego_params.time_begin = k
            ego_vehicle.draw(rnd, draw_params=ego_params)
            rnd.render()
            # rnd.ax.set_title(title)
            filename = 'pgf/' + task[0] +'_frame_' +str(k) + '.pgf'
            rnd.f.savefig(filename, bbox_inches='tight')
            plt.close()
        if make_tikz:
            plt.figure()
            rnd = MPRenderer(plot_limits=task[1],figsize=(task[3][0], task[3][1]))
            if moresteps:            
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1dd0ea"
                rnd.draw_params.time_begin = k+2
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1db0ea"
                rnd.draw_params.time_begin = k+1
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1d7eea"
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ffa0a0"
                ego_params.time_begin = k +2
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff8080"
                ego_params.time_begin = k +1
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "r"
            rnd.draw_params.time_begin = k
            scenario.draw(rnd)
            ego_params.time_begin = k
            ego_vehicle.draw(rnd, draw_params=ego_params)
            rnd.render()
            # rnd.ax.set_title(title)
            code = tikzplotlib.get_tikz_code(figure=rnd.f)
            if add_captions:
                code = code + '\n\caption{' + title + '}' 
            filename = 'tikz/' + task[0] +'_frame_' +str(k) + '.tikz'
            with open(filename, "w") as f:
                f.write(code)
            plt.close()
        
                
            
    # %% Creating plots  tim
    # get EV trajectory and create commonroad object (using DynamicObstacle)
    ego_vehicle_comparison = DynamicObstacle(obstacle_id=id_ev, obstacle_type=ObstacleType.CAR,
                                      obstacle_shape=ev_comparison._shape, initial_state=ev_comparison._initial_state,
                                      prediction=ev_comparison.get_prediction())


    ego_params_comparison = DynamicObstacleParams()
    ego_params_comparison.vehicle_shape.occupancy.shape.facecolor = "r"
    text = controller.applied_method_comparison.copy()
    text.insert(0, 'Initial state')
    #   plot scenario and EV for each time step
    for k in task[2]:           
        plt.figure()
        rnd = MPRenderer(plot_limits=task[1],figsize=(25,10))
        if moresteps:            
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1dd0ea"
            rnd.draw_params.time_begin = k+2
            scenario.draw(rnd)
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1db0ea"
            rnd.draw_params.time_begin = k+1
            scenario.draw(rnd)
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1d7eea"
            ego_params_comparison.vehicle_shape.occupancy.shape.facecolor = "#ffa0a0"
            ego_params_comparison.time_begin = k +2
            ego_vehicle_comparison.draw(rnd, draw_params=ego_params_comparison)
            ego_params_comparison.vehicle_shape.occupancy.shape.facecolor = "#ff8080"
            ego_params_comparison.time_begin = k +1
            ego_vehicle_comparison.draw(rnd, draw_params=ego_params_comparison)
            ego_params_comparison.vehicle_shape.occupancy.shape.facecolor = "r"
        rnd.draw_params.time_begin = k
        scenario.draw(rnd)
        ego_params_comparison.time_begin = k
        ego_vehicle_comparison.draw(rnd, draw_params=ego_params_comparison)
        rnd.render()
        title = "Iteration " + str(k) + ": " + text[k]
        rnd.ax.set_title(title)
        output_text = output_text + title + '\n'
        filename = 'png/' + task[0] +'_frame_' +str(k) +  '_comparison'+ '.png'
        rnd.f.savefig(filename, bbox_inches='tight')
        plt.close()
        if make_pgf:
            plt.figure()
            rnd = MPRenderer(plot_limits=task[1],figsize=(task[3][0], task[3][1]))
            if moresteps:            
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1dd0ea"
                rnd.draw_params.time_begin = k+2
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1db0ea"
                rnd.draw_params.time_begin = k+1
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1d7eea"
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ffa0a0"
                ego_params.time_begin = k +2
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff8080"
                ego_params.time_begin = k +1
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "r"
            rnd.draw_params.time_begin = k
            scenario.draw(rnd)
            ego_params_comparison.time_begin = k
            ego_vehicle_comparison.draw(rnd, draw_params=ego_params_comparison)
            rnd.render()
            # rnd.ax.set_title(title)
            filename = 'pgf/' + task[0] +'_frame_' +str(k) +  '_comparison'+ '.pgf'
            rnd.f.savefig(filename, bbox_inches='tight')        
            plt.close()
        if make_tikz:
            plt.figure()
            rnd = MPRenderer(plot_limits=task[1],figsize=(task[3][0], task[3][1]))
            if moresteps:            
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1dd0ea"
                rnd.draw_params.time_begin = k+2
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1db0ea"
                rnd.draw_params.time_begin = k+1
                scenario.draw(rnd)
                rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#1d7eea"
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ffa0a0"
                ego_params.time_begin = k +2
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff8080"
                ego_params.time_begin = k +1
                ego_vehicle.draw(rnd, draw_params=ego_params)
                ego_params.vehicle_shape.occupancy.shape.facecolor = "r"
            rnd.draw_params.time_begin = k
            scenario.draw(rnd)
            ego_params_comparison.time_begin = k
            ego_vehicle_comparison.draw(rnd, draw_params=ego_params_comparison)
            rnd.render()
            # rnd.ax.set_title(title)
            code = tikzplotlib.get_tikz_code(figure=rnd.f)
            if add_captions:
                code = code + '\n\caption{' + title + '}' 
            filename = 'tikz/' + task[0] +'_frame_' +str(k) + '_comparison'+ '.tikz'
            with open(filename, "w") as f:
                f.write(code)            
            plt.close()

if make_pgf:
    with open("pgf/output.txt", "w") as text_file:
        text_file.write(output_text)
        
if make_tikz:
    with open("tikz/output.txt", "w") as text_file:
        text_file.write(output_text)
