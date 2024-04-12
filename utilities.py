import numpy as np
from scipy.stats import norm
import math
import cmath
from scipy.integrate import odeint
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_dc.geometry.util import chaikins_corner_cutting, resample_polyline
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt

rlar = 90
rclose = 10
rllm = 10

def extract_trajectory(o):
    time_step_list = [o.initial_state.time_step]
    state_list = [project_state_pm(o._initial_state)]
    for state in o._prediction.trajectory.state_list:
        time_step_list.append(state.time_step)
        state_list.append(project_state_pm(state))
    #   check data point in trajectory are properly ordered
    if time_step_list[1:]!=[*range(o._prediction._trajectory._initial_time_step, o._prediction._final_time_step+1)]:
        print('Error: Data point of trajectory of obstacle were not ordered')
        breakpoint()
    return (time_step_list, state_list)

""""
    assumes commonroad framework as input, and [x, vx, y, vy] as output
"""
def project_state_pm(state):
    return np.array([state.position[0], state.velocity*math.cos(state.orientation),
                      state.position[1], state.velocity*math.sin(state.orientation)])

""""
    assumes commonroad framework as input, and [s, d, phi, v] as output
"""
def project_state_kb(state):
    return np.array([state.position[0], state.position[1],
                      state.orientation, state.velocity])

def get_frenet_frame(scenario, planning_problem, n_extra_pts=300, n_back_points=10):
    # based on https://commonroad.in.tum.de/docs/commonroad-drivability-checker/sphinx/06_curvilinear_coordinate_system.html
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    candidate_holder = route_planner.plan_routes()
    route = candidate_holder.retrieve_first_route()
    ref_path = route.reference_path
    #   the chaikin's corner-cutting algorithm creates numerical problems
    # for i in range(0, 10):
    #     ref_path = chaikins_corner_cutting(ref_path)
    ref_path = resample_polyline(ref_path, 2.0)
    
    #   average direction reference pure path (before extending) needed for lane classification
    dir_ref_path = np.mean(ref_path[1:, :]-ref_path[:-1, :], axis=0)
    dir_ref_path = dir_ref_path/np.linalg.norm(dir_ref_path)   #   unit vector
    
    # Extend reference path forward (added by Tommaso to extend extent curv cosy. It is not a real path forward, just approximated, but it's okay for the scope of the scenario)
    # ref_mean_step = np.mean(ref_path[-n_back_points:, :]-ref_path[-n_back_points-1:-1, :], axis=0).reshape(1, -1)
    # new_points = ref_path[-1, :] + ref_mean_step*np.arange(1, n_extra_pts+1).reshape(-1, 1)
    new_points = ref_path[-1, :] + dir_ref_path*np.arange(1, n_extra_pts+1).reshape(-1, 1)
    ref_path = np.vstack((ref_path, new_points))
    
    #   extend reference path backward (added by Tommaso to extend extent curv cosy. It is not a real path backward, just approximated, but it's okay for the scope of the scenario)
    # ref_mean_step = np.mean(ref_path[1:n_back_points+1, :]-ref_path[:n_back_points, :], axis=0).reshape(1, -1)
    # new_points = ref_path[0, :] - ref_mean_step*np.arange(n_extra_pts, 0, -1).reshape(-1, 1)
    new_points = ref_path[0, :] - dir_ref_path*np.arange(n_extra_pts, 0, -1).reshape(-1, 1)
    ref_path = np.vstack((new_points, ref_path))
    
    return CurvilinearCoordinateSystem(ref_path, 50.0, 0.1), ref_path, dir_ref_path

def get_lane_info(scenario, curv_cosy, x0_ev_curv, ref_path = None, planning_problem_set=None, dir_ref_path = None):
    lanelets = scenario._lanelet_network._lanelets
    first_lane = True
    d_min, d_max = 0.0, 0.0   #   min and max bounds road
    for lane in lanelets.values():
        lv_list = lane._left_vertices
        rv_list = lane._right_vertices
        kstp_u = min(10, lv_list.shape[0]-1)    #   used for upper bound
        kstp_l = min(10, lv_list.shape[0])      #   used for lower bound
        try:
            #   check if the lane is vertically oriented (wrt reference path)
            #   use inner product formula with unit vector dir_ref_path
            dir_lane = lv_list[-kstp_l]-lv_list[kstp_u]
            #   0 if horizontal (wrt reference path), 1 if vertical
            lane_orient = np.round(np.arccos(dir_ref_path@dir_lane/np.linalg.norm(dir_lane))/np.pi*2)%2
            if lane_orient==1: #   skip vertical lanes (wrt reference path)
                continue
            
            #   worst case p0 is later than pN, but works anyway for directions and so
            p0l_curv = curv_cosy.convert_to_curvilinear_coords(lv_list[kstp_u][0], lv_list[kstp_u][1])
            pNl_curv = curv_cosy.convert_to_curvilinear_coords(lv_list[-kstp_l][0], lv_list[-kstp_l][1])
            p0r_curv = curv_cosy.convert_to_curvilinear_coords(rv_list[kstp_u][0], rv_list[kstp_u][1])
            pNr_curv = curv_cosy.convert_to_curvilinear_coords(rv_list[-kstp_l][0], rv_list[-kstp_l][1])
            
            if first_lane:
                w_lane = np.linalg.norm(p0l_curv-p0r_curv)
                first_lane = False
            d_min = min(d_min, p0r_curv[1], pNr_curv[1])
            d_max = max(d_max, p0l_curv[1], pNl_curv[1])
        except: #continue
            print('\nProblem curvilinear cosy, Tommaso must fix it\n')
            breakpoint()
            # debugging: print position of point and proj domain border and road map and ref path
            rnd = MPRenderer(figsize=(25, 10))
            scenario.draw(rnd)
            planning_problem_set.draw(rnd)
            rnd.render()
            plt.plot(ref_path[:, 0], ref_path[:, 1], zorder=100, linewidth=2, color='green')
            proj_domain_border = np.asarray(curv_cosy.projection_domain())
            plt.plot(proj_domain_border[:, 0], proj_domain_border[:, 1], zorder=100, color='orange')
            lanelet_borders = np.array([lv_list[kstp_u], lv_list[-kstp_l],
                                        rv_list[-kstp_l], rv_list[kstp_u]])
            plt.plot(lanelet_borders[:, 0], lanelet_borders[:, 1], zorder=100, color='black')
            plt.show(block=True)
            breakpoint()
    if d_max-d_min<0.5:
        print('\nProblem curvilinear cosy, Tommaso must fix it\n')
        breakpoint()
    return {'wlane': w_lane, 'd_min': d_min, 'd_max': d_max}

def lane_pos(x0_ev, x0_tv, lane_info):
    lane_ev = np.round(x0_ev[1]/lane_info['wlane'])
    lane_tvs = []
    for x0_tvi in x0_tv:
        lane_tvs.append(np.round(x0_tvi[2]/lane_info['wlane']))
    return lane_ev, lane_tvs

def lane_center(y0, lane_info):
        return lane_info['wlane']*np.round(y0/lane_info['wlane'])

def curv_ev(x0, curv_cosy, ds=0.1, scenario=None, planning_problem_set=None, ref_path = None):
    try:
        x0_proj = curv_cosy.convert_to_curvilinear_coords(x0[0], x0[1])
        t_vec = curv_cosy.convert_to_cartesian_coords(x0_proj[0]+ds, 0)\
                -curv_cosy.convert_to_cartesian_coords(x0_proj[0]-ds, 0)
    except:
        print('\nProblem curvilinear cosy, Tommaso must fix it\n')
        breakpoint()
        # debugging: print position of point and proj domain border and road map and ref path
        rnd = MPRenderer(figsize=(25, 10))
        scenario.draw(rnd)
        planning_problem_set.draw(rnd)
        rnd.render()
        plt.plot(ref_path[:, 0], ref_path[:, 1], zorder=100, linewidth=2, color='green')
        proj_domain_border = np.asarray(curv_cosy.projection_domain())
        plt.plot(proj_domain_border[:, 0], proj_domain_border[:, 1], zorder=100, color='orange')
        plt.plot(x0[0], x0[1], zorder=100, color='black')
        plt.show(block=True)
        breakpoint()
    theta = np.arctan2(t_vec[1], t_vec[0])
    x0_curv = x0+0  #   do not change initial param
    x0_curv[:2] = x0_proj
    x0_curv[2]-=theta
    return x0_curv

def curv_tv(x0_tv, curv_cosy, ds=0.1):
    try:
        x0_proj = curv_cosy.convert_to_curvilinear_coords(x0_tv[0], x0_tv[2])
        t_vec = curv_cosy.convert_to_cartesian_coords(x0_proj[0]+ds, 0)\
                -curv_cosy.convert_to_cartesian_coords(x0_proj[0]-ds, 0)
    except:
        x0_proj = np.array([1000, 0])# put it so far ahead that it is ingored
        t_vec = np.array([0, 1]) # random number, ignored
    theta = np.arctan2(t_vec[1], t_vec[0])
    phi = np.arctan2(x0_tv[3], x0_tv[1])
    phi -= theta
    vnorm = np.sqrt(x0_tv[1]**2+x0_tv[3]**2)
    x0_tv_curv = x0_tv+0  #   do not change initial param
    x0_tv_curv[0:4:2] = x0_proj
    x0_tv_curv[1] = vnorm*np.cos(phi)
    x0_tv_curv[3] = vnorm*np.sin(phi)
    return x0_tv_curv

"""
     nonlinear method of AV dynamics
"""
def integrate_nonlin_dynamics(x, u, T, lr, lflr_ratio, n_steps=100):
    x_seq = odeint(nl_bicycle_dynamics, x, np.linspace(0, T, n_steps), args=(u, lr, lflr_ratio))
    return np.array(x_seq[-1, :])

"""
     nonlinear bicycle model for real state update in cartesian coordinates
"""
def nl_bicycle_dynamics(x, t, u, lr, lflr_ratio):
    al_ = np.arctan(lflr_ratio*np.tan(u[1]))
    theta = al_+x[2]
    return np.array([x[3]*math.cos(theta),
                    x[3]*math.sin(theta),
                    x[3]*math.sin(al_)/lr,
                    u[0]])
    
    
    
"""
    numerically evaluates matrices of linearized discretized bicycle model in Frenet coordinates
"""
def curved_bicycle_matrices(x0, T, lf, lr, k, tol=1e-5):
    d = x0[1]
    v = x0[3]
    c0 = np.cos(x0[2])
    s0 = np.sin(x0[2])
    alpha1 = lr/(lr+lf)
    
    w0 = 1-k*d
    w1 = 1/w0
    l = v*k*w1

    z0 = cmath.sqrt(1-5*c0**2)
    z1 = s0+z0
    z2 = s0-z0
    e1 = np.exp(l*z1*T/2)
    e2 = np.exp(l*z2*T/2)
    z3 = e1-e2
    z4 = z1*e2-z2*e1
    z5 = z1*e1-z2*e2

    if np.abs(z0)<tol:
        z6 = l*T
        z7 = 2.0
        z8 = 2*(1+l*s0*T)
        z9 = l*T**2/2
        z10 = 2*T
        z11 = 2*T+l*s0*T**2
    else:
        z6 = z3/z0
        z7 = z4/z0
        z8 = z5/z0
        if np.abs(l)<tol:
            z9 = 0.0
            z10 = 2*T
            z11 = 2*T+l*s0*T**2
        else:
            if np.abs(z1*z2)<tol:
                if np.abs(z1)<tol:
                    z9 = (T-2*(e2-1)/l/z2)/z0
                    z10 = -z2/z0*T
                else:
                    z9 = (2*(e1-1)/l/z1-T)/z0
                    z10 = z1/z0*T
            else:
                z9 = 2*((e1-1)/z1-(e2-1)/z2)/z0/l
                z10 = 2*((e2-1)*z1/z2-(e1-1)*z2/z1)/z0/l
            z11 = 2*z6/l

    if np.abs(l)<tol:
        a13 = -v*s0*T
        a14 = T*c0*w1
        a23 = v*c0*T
        a24 = T*s0
        a34 = -k*c0*T*w1
        b11 = c0*T**2/2*w1
        b21 = s0*T**2/2
        b31 = -k*T**2*c0/2*w1
        b12 = -(1+v*T/2/lr)*v*alpha1*s0*T*w1
        b22 = (1+v*T/2/lr)*v*T*c0*alpha1
        b32 = v*alpha1*T/lr
    else:
        if np.abs(c0)<tol:
            a14 = 0.0
            a24 = s0*T
            a34 = 0.0
            b11 = 0.0
            b21 = s0*T**2/2
            b31 = 0.0
        else:
            a14 = (s0*(1-z8/2)+z6)/(v*k*c0)
            a24 = (-1+s0*c0**2*z6+z7/2)/l/c0**2
            a34 = (s0*(z8/2-1)-z6)/v/c0
            b11 = (s0*(T-z11/2)+z9)/(v*k*c0)
            b21 = (-T+s0*c0**2*z9+z10/2)/l/c0**2
            b31 = (s0*(z11/2-T)-z9)/v/c0
        a13 = (1-z8/2)/k
        a23 = w0*c0*z6/k
        w2 = (w0/k/lr+s0)
        b12 = (-T*s0+c0**2*z9+(T-z11/2)*w2)*v*alpha1*w1
        b22 = (z10/2+z9*w2)*v*c0*alpha1
        b32 = (z11*w2/2-z9*c0**2)*l*alpha1
    

    A = np.array([[1.0, c0*w1*z6, a13, a14], 
                    [0.0, z7/2, a23, a24],
                    [0.0, -k*c0*z6*w1, z8/2, a34],
                    [0.0, 0.0, 0.0, 1.0]])

    B = np.array([[b11, b12],
                    [b21, b22],
                    [b31, b32],
                    [T, 0.0]])
      
    q1 = 1/(1-k*x0[1])
    q2 = np.cos(x0[2])
    q3 = x0[3]*q2*q1
    fc = np.array([q3, x0[3]*np.sin(x0[2]), -k*q3, 0])
    return {"A": A.real, "B": B.real, "fc": fc.real}


def compute_lqr(A, B, Q, R, n_iter=4000, tol=1e-4):
    P = 10+Q
    for h in range(n_iter):
        P_ = P
        P = (A.T.dot(P)).dot(A)+Q-((A.T.dot(P)).dot(B)).dot(
                np.linalg.inv((B.T.dot(P)).dot(B)+R)).dot((B.T.dot(P)).dot(A))
        if np.linalg.norm(P-P_)<tol:
                break
    K = -np.linalg.inv((B.T.dot(P)).dot(B)+R).dot(B.T.dot(P)).dot(A)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if np.abs(K[i, j])<0.01:
                K[i, j] = 0
    return K

def truncated_gaussian(mu, sigma, a, b):
    mu_new = np.zeros(a.shape)
    sigma_new = np.zeros(a.shape)
    for k in range(len(mu)):
        alpha = (a[k]-mu[k])/sigma[k]
        beta = (b[k]-mu[k])/sigma[k]
        phia = norm.pdf(alpha)
        phib = norm.pdf(beta)
        Phia = norm.cdf(alpha)
        Phib = norm.cdf(beta)
        dphi = phib-phia
        dPhi = Phib-Phia
        ratiod = dphi/dPhi
        mu_new[k] = mu[k]-sigma[k]*ratiod
        sigma_new[k] = sigma[k]**2*(1-(beta*phib-alpha*phia)/dPhi-ratiod**2)
    return mu_new, sigma_new

def determine_smpc_case(x0_ev, x0_tv, xx_tv, lane_info, N, T, lwo):
    lane_ev, lane_tv = lane_pos(x0_ev, x0_tv, lane_info)
    cases = []
    for i in range(len(x0_tv)):
        casei = 'A' # no constraints
        ds = x0_ev[0]-x0_tv[i][0]
        dv = x0_ev[3]-x0_tv[i][1]
        fclose = rclose+np.max(dv*N*T, 0)
        if -rlar<=ds<=rlar:
            if ds <= -fclose:  casei = 'B'
            # if fclose <= ds:  casei = 'C' #   from Tim, but makes more sense with ds>=0 in general
            if 0 <= ds:  casei = 'C'
            if -fclose <= ds <= 0 and lane_ev == lane_tv[i]:    casei = 'D'
            if lane_ev+1 == lane_tv[i]:
                if -fclose <= ds and x0_ev[0]+0.5*lwo[i][1]+rllm <= x0_tv[i][0]:
                    if dv>=0:   casei = 'E'
                    else:   casei = 'E2'
                if -fclose <= ds <= 0 and x0_ev[0]+0.5*lwo[i][1]+rllm >= x0_tv[i][0]:
                    casei = 'E3'
            if -fclose <= ds <= fclose and lane_ev > lane_tv[i]:  casei = 'F'
            if -fclose <= ds <= 0 and lane_ev+2 <= lane_tv[i]:    casei = 'G'
            # if 0 <= ds <= fclose and lane_ev < lane_tv[i]:    #   removed ds<fclose to activate lateral constraint, Tommaso, June 28
            if 0 <= ds and lane_ev < lane_tv[i]:
                casei = 'H'
        cases.append(casei)
    if 'G' in cases or 'H' in cases:
        cases = [item if item != 'D' and item != 'E' else 'B' for item in cases]
    return cases

def generate_smpc_constraints(x0_ev, x0_tv, xx_tv, cases, N, cornerso, cornersev):
    qs, qd, qt = [[] for k in range(N)], [[] for k in range(N)], [[] for k in range(N)]
    for i in range(len(xx_tv)):
        if cases[i] == 'B':
            for k in range(N):
                qs[k].append(1.0)
                qd[k].append(0.0)
                qt[k].append(-(xx_tv[i][k][0]+cornerso[i][1][k][0]))
        elif cases[i] == 'C':
            for k in range(N):
                qs[k].append(-1.0)
                qd[k].append(0.0)
                qt[k].append(xx_tv[i][k][0]+cornerso[i][0][k][0])
        elif cases[i] == 'D' or cases[i] == 'E':
            for k in range(N):
                qs[k].append(max(0.0, (x0_ev[1]+cornersev[3][1]-(xx_tv[i][k][2]+cornerso[i][1][k][1]))/(x0_ev[0]+cornersev[3][0]-(xx_tv[i][k][0]+cornerso[i][1][k][0]))))
                qd[k].append(-1.0)
                qt[k].append(xx_tv[i][k][2]+cornerso[i][1][k][1]-qs[k][-1]*(xx_tv[i][k][0]+cornerso[i][1][k][0]))
        elif cases[i] == 'E2':
            for k in range(N):
                qs[k].append(1.0)
                qd[k].append(0.0)
                qt[k].append(-(xx_tv[i][k][0]+cornerso[i][1][k][0]))
        elif cases[i] == 'E3' or cases[i] == 'G':
            for k in range(N):
                qs[k].append(0.0)
                qd[k].append(1.0)
                qt[k].append(-(xx_tv[i][k][2]+cornerso[i][2][k][1]))
        elif cases[i] == 'F':
            for k in range(N):
                qs[k].append(0.0)
                qd[k].append(-1.0)
                qt[k].append(xx_tv[i][k][2]+cornerso[i][1][k][1])
        elif cases[i] == 'H':
            for k in range(N):
                qs[k].append(0.0)
                qd[k].append(1.0)
                qt[k].append(-(xx_tv[i][k][2]+cornerso[i][3][k][1]))
        else:   #   A or J
            if cases[i] != 'A' and cases[i] != 'J':
                print('Error: handleded case smpc')
                breakpoint()
            for k in range(N):
                qs[k].append(0.0)
                qd[k].append(0.0)
                qt[k].append(0.0)
    return {'s': np.asarray(qs), 'd': np.asarray(qd), 't': np.asarray(qt)}

def determine_cvpm_case(x0_ev, x0_tv, xx_tv, lane_info, N, T, lwo):
    lane_ev, lane_tv = lane_pos(x0_ev, x0_tv, lane_info)
    cases = []
    for i in range(len(x0_tv)):
        casei = 'A' # no constraints
        ds = x0_ev[0]-x0_tv[i][0]
        dv = x0_ev[3]-x0_tv[i][1]
        dlane = lane_ev-lane_tv[i]
        fclose = rclose+np.max(dv*N*T, 0)
        if -rlar<=ds<=rlar:
            if ds<-fclose:      casei = 'B'
            if 0<=ds<=fclose:
                if dlane>0:     casei = 'Fa'
                elif dlane<0:   casei = 'Ha'
                #   no constraints (aka A) if dlane==0 as it cannot happen in practice
            if -fclose<=ds<=0:
                if dlane==0:    casei = 'D'
                elif dlane>0: #   NB y_lane_tv = lane_tv[i]*lane_info['wlane']
                    if x0_ev[1]>=(lane_tv[i]+0.5)*lane_info['wlane']+0.5*lwo[i][1]: casei = 'Fb' #  geq!
                    else:   casei = 'F2'
                else:   #   dlane<0
                    if x0_ev[1]<=(lane_tv[i]+0.5)*lane_info['wlane']+0.5*lwo[i][1]: casei = 'Hb' #  leq!
                    else:   casei = 'H2'
            if ds>0 and dlane==0:
                casei = 'C'
        cases.append(casei)
    return cases

def generate_cvpm_constraints(x0_ev, x0_tv, xx_tv, cases, N, cornerso, cornersoexp, cornersovar, cornersev, lane_info):
    qs, qd, qt = [[] for k in range(N)], [[] for k in range(N)], [[] for k in range(N)]
    qtexp = [[] for k in range(N)]
    qtvar = [[] for k in range(N)]
    for i in range(len(xx_tv)):
        if cases[i] == 'B' or cases[i] == 'D' or cases[i] == 'F2' or cases[i] == 'H2':
            for k in range(N):
                qs[k].append(1.0)
                qd[k].append(0.0)
                qt[k].append(-(cornerso[i][1][k][0]))
                qtexp[k].append(-cornersoexp[i][1][k][0])
                qtvar[k].append(cornersovar[i][1][k][0])
        elif cases[i] == 'Fa' or cases[i] == 'Fb':
            for k in range(N):
                qs[k].append(0.0)
                qd[k].append(-1.0)
                qt[k].append(cornerso[i][1][k][1])
                qtexp[k].append(cornersoexp[i][1][k][1])
                qtvar[k].append(cornersovar[i][1][k][1])
        elif cases[i] == 'Ha' or cases[i] == 'Hb':
            for k in range(N):
                qs[k].append(0.0)
                qd[k].append(1.0)
                qt[k].append(-(cornerso[i][2][k][1]))
                qtexp[k].append(-cornersoexp[i][2][k][1])
                qtvar[k].append(cornersovar[i][2][k][1])
        elif cases[i] == 'C':
            for k in range(N):
                qs[k].append(-1.0)
                qd[k].append(0.0)
                qt[k].append(cornerso[i][0][k][0])
                qtexp[k].append(cornersoexp[i][0][k][0])
                qtvar[k].append(cornersovar[i][0][k][0])
        else:   #   A
            if cases[i] != 'A':
                print('Error: handleded case cvpm')
                breakpoint()
            for k in range(N):
                qs[k].append(0.0)
                qd[k].append(0.0)
                qt[k].append(0.0)
                qtexp[k].append(0.0)
                qtvar[k].append(1e-6)
    if (np.asarray(qtvar)<0).any():
        breakpoint()
    return {'s': np.asarray(qs), 'd': np.asarray(qd), 't': np.asarray(qt),
            'texp': np.asarray(qtexp), 'tvar': np.asarray(qtvar)}

def check_if_equal(constr, constr2):
    equals = True
    for aa in constr:
        equali = False
        for bb in constr2:
            if str(aa)==str(bb):
                equali = True
                break
        if equali == False:
            equals = False

    for bb in constr2:
        equali = False
        for aa in constr:
            # if str(bb)=='var1[0:4, 0] == [15.0003  0.      0.     22.    ]':
            #     print(str(aa))
            if str(aa)==str(bb):
                equali = True
                break
        if equali == False:
            equals = False
    print(equals)