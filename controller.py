import numpy as np
import cvxpy as cvx
from utilities import curved_bicycle_matrices, truncated_gaussian, determine_smpc_case, generate_smpc_constraints, determine_cvpm_case, generate_cvpm_constraints, integrate_nonlin_dynamics, check_if_equal
import matplotlib.pyplot as plt

# %% Select if you want plots while running the controller
plot_debug = False
# plot_debug = True
plot_online = False
# plot_online = True


#--------------------------------------------
# %% SMPCVPM class
class SMPCVPM:
    def __init__(self, T, N, Ao, Bo, Ko, umaxo, umino, P_w, P_v, lwo, v_amp, lwev, vehicle_params,
                 vmaxev, vminev, umaxev, uminev, dumaxev, xref_ev=np.array([0, 0, 0, 27]),
                 Q_ev=np.diag([0, 1, 1, 1]), R_ev=np.diag([1, 1]), beta=0.9, features_store=None):
        self._T = T
        self._N = N
        self._u_guess = np.zeros((2, N))
        self._Ao = Ao
        self._Bo = Bo
        self._Ko = Ko
        self._umaxo = umaxo
        self._umino = umino
        self._lwo = lwo
        self._v_amp = v_amp
        self._PP_o = SMPCVPM.prediction_covariance(Ao, Bo, Ko, N, P_w, P_v)
        self._cornerso = SMPCVPM.obtain_corners_o(
            N, lwo, beta, self._PP_o)
        self._cornersev = [np.array([lwev[0]/2, lwev[1]/2]),  # front left
                           np.array([-lwev[0]/2, lwev[1]/2]),  # rear left
                           np.array([-lwev[0]/2, -lwev[1]/2]),  # rear right
                           np.array([lwev[0]/2, -lwev[1]/2])]  # front right
        self._lr = vehicle_params.l/2
        self._lf = self._lr
        self._lflr_ratio = 0.5
        self._uminev = uminev
        self._u_prev = 0*umaxev
        self._u_prev_comparison = 0*umaxev
        self._u_backup_comparison = np.zeros((umaxev.shape[0], self._N))   #   length might also differ from horizon length, but it is convenient for implementation reasons
        self._dumax = dumaxev
        self._xref_ev = xref_ev
        self._Q_ev = Q_ev
        self._R_ev = R_ev
        self._x_smpc = cvx.Variable(shape=(xref_ev.shape[0], self._N+1))
        self._x_smpc_comparison = cvx.Variable(shape=(xref_ev.shape[0], self._N+1))
        self._u_smpc = cvx.Variable(shape=(2, self._N))
        self._u_smpc_comparison = cvx.Variable(shape=(2, self._N))
        self._cost_smpc = 0
        self._cost_smpc_comparison = 0
        self._constr_smpc = [self._u_smpc[0, :-1]-self._u_smpc[0, 1:] <= dumaxev[0],  # missing input k=0 wrt previous!
                             self._u_smpc[0, 1:] - self._u_smpc[0, :-1] <= dumaxev[0],
                             self._u_smpc[1, :-1] - self._u_smpc[1, 1:] <= dumaxev[1],
                             self._u_smpc[1, 1:] - self._u_smpc[1, :-1] <= dumaxev[1],
                             self._u_smpc[0, :] <= umaxev[0], self._u_smpc[0, :] >= uminev[0],
                             self._u_smpc[1, :] <= umaxev[1], self._u_smpc[1, :] >= uminev[1],
                             self._x_smpc[3, 1:] <= vmaxev, self._x_smpc[3, 1:] >= vminev]
        self._constr_smpc_comparison = [self._u_smpc_comparison[0, :-1]-self._u_smpc_comparison[0, 1:] <= dumaxev[0],  # missing input k=0 wrt previous!
                             self._u_smpc_comparison[0, 1:] - self._u_smpc_comparison[0, :-1] <= dumaxev[0],
                             self._u_smpc_comparison[1, :-1] - self._u_smpc_comparison[1, 1:] <= dumaxev[1],
                             self._u_smpc_comparison[1, 1:] - self._u_smpc_comparison[1, :-1] <= dumaxev[1],
                             self._u_smpc_comparison[0, :] <= umaxev[0], self._u_smpc_comparison[0, :] >= uminev[0],
                             self._u_smpc_comparison[1, :] <= umaxev[1], self._u_smpc_comparison[1, :] >= uminev[1],
                             self._x_smpc_comparison[3, 1:] <= vmaxev, self._x_smpc_comparison[3, 1:] >= vminev]
        self.dumaxev = dumaxev
        self.uminev = uminev
        self.vminev = vminev
        self.umaxev = umaxev
        self.vmaxev = vmaxev

        for k in range(self._N):
            self._cost_smpc += cvx.quad_form(self._x_smpc[:, k+1]-self._xref_ev, self._Q_ev)\
                + cvx.quad_form(self._u_smpc[:, k], self._R_ev)
            self._cost_smpc_comparison += cvx.quad_form(self._x_smpc_comparison[:, k+1]-self._xref_ev, self._Q_ev)\
                + cvx.quad_form(self._u_smpc_comparison[:, k], self._R_ev)
        self._comp_time_smpc = []
        self._comp_time_cvpm1 = []
        self._comp_time_cvpm2 = []
        self._comp_time_isFeasible = [] #   Check saftey
        self._comp_time_case_cvpm = []  #   Case-Decision of CVPM
        self._comp_time_smpc_comparison = []
        self._comp_time_ftp_now_comparison = []
        self._comp_time_ftp_next_comparison = []
        self.counter = 0
        self.counter_comparison = 0
        self.applied_method = []
        self.applied_method_comparison = []
        self.input_sequences = []
        self.input_sequences_comparison = []
        self.stage_cost_sequence = []
        self.stage_cost_sequence_comparison = []
        self.x0_plot = None
        if features_store is not None:
            for key in features_store:
                exec('self.%s=0'%key)


    def run_step(self, x0_ev, x0_tv, xrefo, lane_info, dmaxlane, dminlane):
        self.counter = self.counter + 1
        self.add_iteration_time_counters(comparison=False)
        print("\nController Iteration: " + str(self.counter))
        q, xx_tv, smpc_cases = self.smpc_constraints(x0_ev, x0_tv, xrefo, lane_info, comparison=False)
        evlin = curved_bicycle_matrices(x0_ev, self._T, self._lf, self._lr, 0)
        evlin['C'] = evlin['fc']*self._T+x0_ev
        self._curr_constr_smpc = [self._x_smpc[1, 1:] <= dmaxlane-self._cornersev[0][1],
                                  self._x_smpc[1, 1:] >= dminlane+self._cornersev[0][1]]
        self._curr_constr_smpc.extend(self._constr_smpc)

        self.dmaxlane = dmaxlane
        self.dminlane = dminlane

        u_sol = None
        # SMPC
        isSMPCfeasible = self.solve_smpc(x0_ev, evlin, q, comparison=False)
        x_pred = self._x_smpc.value
        if isSMPCfeasible:
            text = "SMPC is feasible"
            u_smpc = self._u_smpc.value
        else:
            text = "SMPC is not feasible,"
        
            
        title = "SMPC constraint-cases =" + str(smpc_cases) + '\n' + text
        if plot_debug:
            self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=0, xx_tv=xx_tv, x_pred=x_pred, title=title, comparison=False)
            self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=-1, xx_tv=xx_tv, x_pred=x_pred, title=title+' (Terminal State)', comparison=False)
        if plot_online:
            self.plot_feasible_region(q, x0_ev, x0_tv, text=title, comparison=False)
            
        if isSMPCfeasible:
            isSafe, q, cvpm_cases = self.check_smpc_safety(x0_ev, self._u_smpc.value[:, 0], evlin, x0_tv, xrefo, lane_info)
            if isSafe:
                text = text + " and safe."
                u_sol = u_smpc
            else:
                text = text + " but not safe,"
                
            title = "Safty Check constraint-cases =" + str(cvpm_cases) + '\n' + text
            if plot_debug:
                self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=0, title=title, comparison=False)
                self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=-1, title=title+' (Terminal State)', comparison=False)
            if plot_online:
                self.plot_feasible_region(q, x0_ev, x0_tv, text=title, comparison=False)    
                
        # CVPM
        if u_sol is None:
            bool_safe, constr, q, cvpm_cases = self.check_case_cvpm(x0_ev, self._u_prev, evlin, x0_tv, xrefo, lane_info)
            if bool_safe:
                text = text + " use CVPM Safe Case."
                u_sol, status = self.solve_cvpm_safe_case(constr, comparison=False)
                x_pred = self._x_smpc.value
                
                title = "CVPM constraint-cases =" + str(cvpm_cases)+ '\n' + text
                if plot_debug:
                    self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=0, xx_tv=xx_tv, x_pred=x_pred, title=title, comparison=False)
                    self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=-1, xx_tv=xx_tv, x_pred=x_pred, title=title+' (Terminal State)', comparison=False)
                if plot_online:
                    self.plot_feasible_region(q, x0_ev, x0_tv, text=title, comparison=False)
                    
            else:
                text = text + " use CVPM Probabilistic Case."
                u_sol, q, cvpm_cases  = self.solve_cvpm_probabilistic_case(x0_ev, x0_tv, xrefo, self._u_prev, evlin, lane_info)
                x_pred = self._x_smpc.value
            
                title = "CVPM constraint-cases =" + str(cvpm_cases) + '\n' + text
                if plot_debug:
                    self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=0, xx_tv=xx_tv, x_pred=x_pred, title=title, comparison=False)
                    self.plot_feasible_region_sperat(q, x0_ev=x0_ev, x0_tv=x0_tv, n=-1, xx_tv=xx_tv, x_pred=x_pred, title=title+' (Terminal State)', comparison=False)
                if plot_online:
                    self.plot_feasible_region(q, x0_ev, x0_tv, text=text, comparison=False)
                
        print(text)
        self.applied_method.append(text)
        self.input_sequences.append(u_sol)
        self.update_applied_input(u_sol, x0_ev, comparison=False)
        return u_sol[:, 0]
    
    def run_step_comparison(self, x0_ev, x0_tv, xrefo, lane_info, dmaxlane, dminlane):
        self.counter_comparison = self.counter_comparison + 1
        self.add_iteration_time_counters(comparison=True)
        print("\nController Iteration (Comparison): " + str(self.counter_comparison))
        q, xx_tv, smpc_cases = self.smpc_constraints(x0_ev, x0_tv, xrefo, lane_info, comparison=True)
        evlin = curved_bicycle_matrices(x0_ev, self._T, self._lf, self._lr, 0)
        evlin['C'] = evlin['fc']*self._T+x0_ev
        self._curr_constr_smpc_comparison = [self._x_smpc_comparison[1, 1:] <= dmaxlane-self._cornersev[0][1],
                                  self._x_smpc_comparison[1, 1:] >= dminlane+self._cornersev[0][1]]
        self._curr_constr_smpc_comparison.extend(self._constr_smpc_comparison)

        self.dmaxlane = dmaxlane
        self.dminlane = dminlane

        u_sol = None
        # SMPC branch
        if self.solve_smpc(x0_ev, evlin, q, comparison=True):
            text = "SMPC is feasible. Try FTP(x^+):"
            
            #   Try FTP(x^+)
            u_smpc = self._u_smpc_comparison.value
            x_smpc = self._x_smpc_comparison.value
            if plot_debug:
                self.plot_feasible_region_sperat(q, x0_ev, x0_tv,  0, x_pred=x_smpc, comparison=True)
                self.plot_feasible_region_sperat(q, x0_ev, x0_tv, -1, x_pred=x_smpc, comparison=True)
            u_ftp_next = self.solve_ftp_next(x0_ev, self._u_smpc_comparison.value[:, 0],
                                                        evlin, x0_tv, xrefo, lane_info,)
            if u_ftp_next is not None:
                text = text + " feasible."
                u_sol = u_smpc
                self.update_backup(u_new=u_ftp_next,tail=False)
            else:
                text = text + " infeasible. Apply backup."
                u_sol = self._u_backup_comparison.copy()
                self.update_backup()
        else:
            text = "SMPC is not feasible,"
        # FTP branch
        if u_sol is None:
            text = text + " try FTP(x):"
            bool_safe, constr, q, cvpm_cases = self.check_case_cvpm(x0_ev, self._u_prev_comparison, evlin, x0_tv, xrefo, lane_info)
            if bool_safe:
                u_sol, status = self.solve_cvpm_safe_case(constr, comparison=True)
                if status!='optimal':
                    #   apply backup
                    text = text + " infeasible. Apply backup."
                    u_sol = self._u_backup_comparison.copy()
                    self.update_backup()
                else:
                    text = text + " feasible"
                    self.update_backup(u_new=u_sol)
            else:
                #   apply backup
                text = text + " infeasible. Apply backup."
                u_sol = self._u_backup_comparison.copy()
                self.update_backup()

        print(text)
        self.applied_method_comparison.append(text)
        self.input_sequences_comparison.append(u_sol)
        self.update_applied_input(u_sol, x0_ev, comparison=True)
        return u_sol[:, 0]  

    def update_applied_input(self, u, x0, comparison=False):
        curr_stage_cost = (cvx.quad_form(x0-self._xref_ev, self._Q_ev)\
                + cvx.quad_form(u[:, 0], self._R_ev)).value
        if comparison:
            self._u_prev_comparison = u[:, 0]
            self._u_guess_comparison = np.hstack((u[:, 1:], 0*self._u_prev_comparison.reshape([u.shape[0], 1])))
            self.stage_cost_sequence_comparison.append(curr_stage_cost)
        else:
            self._u_prev = u[:, 0]
            self._u_guess = np.hstack((u[:, 1:], 0*self._u_prev.reshape([u.shape[0], 1])))
            self.stage_cost_sequence.append(curr_stage_cost)
    
    #   stores new backup (either u_new or tail u_new or tail current backup)
    def update_backup(self, u_new=None, tail=True):
        if tail:
            if u_new is None:
                u_new = self._u_backup_comparison
            self._u_backup_comparison = np.hstack((u_new[:, 1:], 0*u_new[:, 0].reshape((-1, 1))))
        else:
            self._u_backup_comparison = u_new
            

    def check_smpc_safety(self, x0_ev, u_prev, evlin_prev, x0_tv, xrefo, lane_info):
        """
        compute the next state x1 and evaluate if there exists a solution

        Parameters
        ----------
        x0_ev : np.ndarray,     current state of the ego vehicle
        u_prev : np.ndarray,    previous Input for computing the next state
        evlin : function,       dynamic of car
        x0_tv : np.ndarray,     current positions of the target vehicles
        xrefo : np.ndarray,     target vehicles reference state

        Returns
        -------
        TYPE
        Bool, Ture if there exists a solution in the next step, False: there exists no solution, i.e. Empty Set of constraints

        """
        # constraints CVPM robust step 1
        q_cvpm, _ = self.cvpm_constraints(x0_ev, x0_tv, xrefo, lane_info, comparison=False)

        # Compute next step with ode of nonlinear system (is done only once)
        x1 = integrate_nonlin_dynamics(x0_ev, u_prev, self._T, self._lr, self._lflr_ratio)

        # check if SMPC solution is robust at first predicted step
        one_step_check = x1[0]*q_cvpm['s'][0, :]+x1[1]*q_cvpm['d'][0, :]+q_cvpm['t'][0, :]
        if any(one_step_check > 0):
            smpc_one_step_robust = False
        else:
            smpc_one_step_robust = True

        
        evlin = curved_bicycle_matrices(x1, self._T, self._lf, self._lr, 0)
        evlin['C'] = evlin['fc']*self._T+x1
        x1_tv = []
        for i in range(len(x0_tv)):
            uik = np.minimum(self._umaxo[i], np.maximum(
                self._umino[i], self._Ko[i]@(x0_tv[i]-xrefo[i])))
            x1_tv.append(self._Ao@x0_tv[i]+self._Bo@uik)

        q, cvpm_cases = self.cvpm_constraints(x1, x1_tv, xrefo, lane_info, comparison=False)
        
        return smpc_one_step_robust and self.isFeasible(x1, u_prev, evlin, q, comparison=False, check_safety=True)[0], q, cvpm_cases
    
    def solve_ftp_next(self, x0_ev, u_prev, evlin_prev, x0_tv, xrefo, lane_info):
        """
        compute the next state x1 and the FTP solution if it exists

        Parameters
        ----------
        x0_ev : np.ndarray,     current state of the ego vehicle
        u_prev : np.ndarray,    previous Input for computing the next state
        evlin : function,       dynamic of car
        x0_tv : np.ndarray,     current positions of the target vehicles
        xrefo : np.ndarray,     target vehicles reference state

        Returns
        -------
        TYPE
        u_sol, None if no solution exists

        """        
        # Compute next step linearly
        x1 = evlin_prev['A']@(x0_ev - x0_ev) + \
                evlin_prev['B']@u_prev + evlin_prev['C']
        
        #   constraints CVPM robust step 1
        q_cvpm, _ = self.cvpm_constraints(x0_ev, x0_tv, xrefo, lane_info, comparison=True)
        
        #   constraints if SMPC solution is robust at step 1
        one_step_check = x1[0]*q_cvpm['s'][0, :]+x1[1]*q_cvpm['d'][0, :]+q_cvpm['t'][0, :]
        if any(one_step_check > 0):            
            return None


        evlin = curved_bicycle_matrices(x1, self._T, self._lf, self._lr, 0)
        evlin['C'] = evlin['fc']*self._T+x1
        x1_tv = []
        for i in range(len(x0_tv)):
            uik = np.minimum(self._umaxo[i], np.maximum(
                self._umino[i], self._Ko[i]@(x0_tv[i]-xrefo[i])))
            x1_tv.append(self._Ao@x0_tv[i]+self._Bo@uik)

        q, cvpm_cases = self.cvpm_constraints(x1, x1_tv, xrefo, lane_info, comparison=True)
        
        constr = [self._x_smpc_comparison[:, 0] == x1,
                  self._u_smpc_comparison[:, 0]-u_prev <= self._dumax,
                  u_prev-self._u_smpc_comparison[:, 0] <= self._dumax]
        constr.extend(self._curr_constr_smpc_comparison)
        for k in range(self._N):
            constr.append(self._x_smpc_comparison[:, k+1] == evlin['A']@(
                self._x_smpc_comparison[:, k]-x1)+evlin['B']@self._u_smpc_comparison[:, k]+evlin['C'])
            constr.append(
                0 >= self._x_smpc_comparison[0, k]*q['s'][k]+self._x_smpc_comparison[1, k]*q['d'][k]+q['t'][k])
        prob = cvx.Problem(cvx.Minimize(self._cost_smpc_comparison), constr)
        try:
            prob.solve()
        except cvx.error.SolverError:
            prob._status = 'failed'
            
        self._comp_time_ftp_next_comparison[-1] = prob._solve_time
                
        if prob._status == 'optimal':
            return self._u_smpc_comparison.value
        else:
            return None

    def smpc_constraints(self, x0_ev, x0_tv, xrefo, lane_info, comparison=False):
        """
        Compute constraints based on the target vehicles for SMPC (different to CVPM constraints)
        for a given state of the ego vehicle and the target vehicle

        Parameters
        ----------
        x0_ev : np.ndarray,     current state of the ego vehicle
        x0_tv : np.ndarray,     current positions of the target vehicles
        xrefo : np.ndarray,     target vehicles reference state
        Returns
        -------
        TYPE
            Dictionary. contains constraints based on the Target vehicles

        """
        xx_tv = self.predict_trajectory_obstacles(x0_tv, xrefo)
        cornerso = self.add_velocity_cornerso(x0_ev, x0_tv)
        smpc_cases = determine_smpc_case(
            x0_ev, x0_tv, xx_tv, lane_info, self._N, self._T, self._lwo)
        q = generate_smpc_constraints(x0_ev, x0_tv, xx_tv, smpc_cases, self._N, cornerso, self._cornersev)
        q['cases'] = smpc_cases
        return q, xx_tv, smpc_cases

    def cvpm_constraints(self, x0_ev, x0_tv, xrefo, lane_info, comparison=False):
        """
        Compute constraints based on the target vehicles for CVPM (different to SMPC constraints)
        for a given state of the ego vehicle and the target vehicle

        Parameters
        ----------
        x0_ev : np.ndarray,     current state of the ego vehicle
        x0_tv : np.ndarray,     current positions of the target vehicles
        xrefo : np.ndarray,     target vehicles reference state
        Returns
        -------
        TYPE
            Dictionary. contains constraints based on the Target vehicles

        """
        xx_tv = self.predict_trajectory_obstacles(x0_tv, xrefo)
        # cornerso = self.add_velocity_cornerso(x0_ev, x0_tv)
        cornerso, cornersoexp, cornersovar = self.get_reachable_sets(x0_tv, xrefo, x0_ev)
        cvpm_cases = determine_cvpm_case(
            x0_ev, x0_tv, xx_tv, lane_info, self._N, self._T, self._lwo)
        q = generate_cvpm_constraints(x0_ev, x0_tv, xx_tv, cvpm_cases, self._N, cornerso,
                                      cornersoexp, cornersovar, self._cornersev, lane_info)
        q['cases'] = cvpm_cases
        return q, cvpm_cases

    def isFeasible(self, x0_ev, u_prev, evlin, q, comparison=False, check_safety=True):
        """
        Function to check if the set of constraints is empty or a feasible solution exists

        Parameters:
        x0_ev : np.ndarray,     current State
        u_prev : np.ndarray,    previous Input for constraint limit in input changes
        evlin : function,       dynamic of car
        q : dictionary,         Constraints based on the Target vehicle.
        -------
        Returns:
        Bool, Ture if there exists a solution, False: there exists no solution, i.e. Empty Set of constraints
        List, List of constraints with
        """
        if comparison:
            x_ = self._x_smpc_comparison
            u_ = self._u_smpc_comparison
            curr_constr = self._curr_constr_smpc_comparison
        else:
            x_ = self._x_smpc
            u_ = self._u_smpc
            curr_constr = self._curr_constr_smpc
        constr = [x_[:, 0] == x0_ev,
                  u_[:, 0]-u_prev <= self._dumax,
                  u_prev-u_[:, 0] <= self._dumax]
        constr.extend(curr_constr)
        for k in range(self._N):
            constr.append(x_[:, k+1] == evlin['A']@(
                x_[:, k]-x0_ev)+evlin['B']@u_[:, k]+evlin['C'])
            constr.append(
                0 >= x_[0, k]*q['s'][k]+x_[1, k]*q['d'][k]+q['t'][k])
        prob = cvx.Problem(cvx.Minimize(1), constr)
        try:
            prob.solve()
        except cvx.error.SolverError:
            prob._status = 'failed'
        if not comparison:
            # self._comp_time_isFeasible.append(prob._solve_time)
            if check_safety:
                self._comp_time_isFeasible[-1] = prob._solve_time
            else:
                self._comp_time_case_cvpm[-1] = prob._solve_time
        
        return (prob._status == 'optimal', constr)

    def solve_smpc(self, x0_ev, evlin, q, comparison=False):
        if comparison:
            x_smpc = self._x_smpc_comparison
            u_smpc = self._u_smpc_comparison
            u_prev = self._u_prev_comparison
            curr_constr = self._curr_constr_smpc_comparison
            cost_smpc = self._cost_smpc_comparison
            # comp_time_list = self._comp_time_smpc_branch_comparison
            comp_time_list = self._comp_time_smpc_comparison
        else:
            x_smpc = self._x_smpc
            u_smpc = self._u_smpc
            u_prev = self._u_prev
            curr_constr = self._curr_constr_smpc
            cost_smpc = self._cost_smpc
            comp_time_list = self._comp_time_smpc
    
        constr = [x_smpc[:, 0] == x0_ev,
                u_smpc[:, 0]-u_prev <= self._dumax,
                u_prev-u_smpc[:, 0] <= self._dumax]
        constr.extend(curr_constr)
        for k in range(self._N):
            constr.append(x_smpc[:, k+1] == evlin['A']@(
                x_smpc[:, k]-x0_ev)+evlin['B']@u_smpc[:, k]+evlin['C'])
            constr.append(
                0 >= x_smpc[0, k+1]*q['s'][k]+x_smpc[1, k+1]*q['d'][k]+q['t'][k])
        prob = cvx.Problem(cvx.Minimize(cost_smpc), constr)
        try:
            prob.solve()
        except cvx.error.SolverError:
            prob._status = 'failed'
        # comp_time_list.append(prob._solve_time)
        comp_time_list[-1] = prob._solve_time
        return prob._status == 'optimal'

    def check_case_cvpm(self, x0_ev, u_prev, evlin, x0_tv, xrefo, lane_info, comparison=False):
        """
        Compute the constraints for the safe case and check if it is a feasible set of constraints
        
        Parameters
        ----------
        x0_ev : np.ndarray,     current State
        u_prev : np.ndarray,    previous Input for constraint limit in input changes
        evlin : function,       dynamic of car
        q : dictionary,         Constraints based on the Target vehicle.
        x0_tv : np.ndarray,     current positions of the target vehicles
        xrefo : np.ndarray,     target vehicles reference state

        Returns
        -------
        bool_safe : boole,      Safe case is feasible
        constr : list,          List of constraints for the safe case
        """
        q, cvpm_cases = self.cvpm_constraints(x0_ev, x0_tv, xrefo, lane_info, comparison=comparison)
        bool_safe, constr = self.isFeasible(x0_ev, u_prev, evlin, q, comparison=comparison, check_safety=False)
        return bool_safe, constr, q, cvpm_cases

    def solve_cvpm_safe_case(self, constr, comparison=False):
        """
        Optimize the cost function based on the constraints

        Parameters
        ----------
        constr : list,          List of constraints for the safe case
        
        Returns
        -------
            array: optimal sequence of inputs 
        """
        if comparison:
            prob = cvx.Problem(cvx.Minimize(self._cost_smpc_comparison), constr)
            try:
                prob.solve()
                return self._u_smpc_comparison.value, prob._status
            except cvx.error.SolverError:
                return self._u_smpc_comparison.value, 'failed'
        else:
            prob = cvx.Problem(cvx.Minimize(self._cost_smpc), constr)
            prob.solve()
            return self._u_smpc.value, prob._status

    def solve_cvpm_probabilistic_case(self, x0_ev, x0_tv, xrefo, u_prev, evlin, lane_info):
        """
        construct set for the probabilistic optimization and find the input sequence of the
        ego vehicle leading to the minimal probability of constraint violation 

        Parameters
        ----------
        x0_ev : np.ndarray,     current State
        x0_tv : np.ndarray,     current positions of the target vehicles
        xrefo : np.ndarray,     target vehicles reference state
        u_prev : np.ndarray,    previous Input for constraint limit in input changes
        evlin : function,       dynamic of car

        Returns
        -------
        TYPE
            array: optimal sequence of inputs 
        """
        q, cvpm_cases = self.cvpm_constraints(x0_ev, x0_tv, xrefo, lane_info, comparison=False)

        x_soft = cvx.Variable(shape=q['t'].shape)
        constr = [self._u_smpc[0, :-1]-self._u_smpc[0, 1:] <= self.dumaxev[0],  # missing input k=0 wrt previous!
                  self._u_smpc[0, 1:] - self._u_smpc[0, :-1] <= self.dumaxev[0],
                  self._u_smpc[1, :-1] - self._u_smpc[1, 1:] <= self.dumaxev[1],
                  self._u_smpc[1, 1:] - self._u_smpc[1, :-1] <= self.dumaxev[1],
                  self._u_smpc[0, :] <= self.umaxev[0],
                  self._u_smpc[0, :] >= self.uminev[0],
                  self._u_smpc[1, :] <= self.umaxev[1],
                  self._u_smpc[1, :] >= self.uminev[1],
                  self._x_smpc[3, 1:] <= self.vmaxev,
                  self._x_smpc[3, 1:] >= self.vminev,
                  self._x_smpc[1, 1:] <= self.dmaxlane-self._cornersev[0][1],
                  self._x_smpc[1, 1:] >= self.dminlane+self._cornersev[0][1],
                  self._x_smpc[:, 0] == x0_ev,
                  self._u_smpc[:, 0]-u_prev <= self._dumax,
                  u_prev-self._u_smpc[:, 0] <= self._dumax,
                  x_soft <= 0]

        for k in range(self._N):
            constr.append(self._x_smpc[:, k+1] == evlin['A']@(
                self._x_smpc[:, k]-x0_ev)+evlin['B']@self._u_smpc[:, k]+evlin['C'])
        Sigma = q['tvar']
        cost = 0
        for k in range(q['t'].shape[1]):
            x = cvx.multiply(q['s'][:, k], self._x_smpc[0, 1:self._N+1]) + \
                cvx.multiply(q['d'][:, k], self._x_smpc[1, 1:self._N+1]) + \
                q['texp'][:, k] - x_soft[:, k]
            cost = cost + cvx.quad_form(x, np.diag(np.reciprocal(Sigma[:, 0])))

        prob = cvx.Problem(cvx.Minimize(cost), constr)
        try:
            prob.solve()
            # self._comp_time_cvpm2.append(prob._solve_time)
            self._comp_time_cvpm2[-1] = prob._solve_time
            self._u_smpc.value[0, :]
        except:
            breakpoint()
            return np.zeros(self._u_smpc.shape)
        return self._u_smpc.value, q, cvpm_cases
    
    def obtain_corners_o(N, lwo, beta, PP_o):
        cornerso = []
        kappa = -2*np.log(1-beta)
        for i in range(len(PP_o)):
            cornersi = [[] for k in range(4)]
            for k in range(N):
                #   kappa must be inside sqrt()
                cornersi[0].append(np.array([lwo[i][0]+np.sqrt(PP_o[i][k][0, 0]*kappa),  # front left
                                            lwo[i][1]+np.sqrt(PP_o[i][k][2, 2]*kappa)]))
                cornersi[1].append(np.array([-lwo[i][0]-np.sqrt(PP_o[i][k][0, 0]*kappa),  # rear left
                                            lwo[i][1]+np.sqrt(PP_o[i][k][2, 2]*kappa)]))
                cornersi[2].append(np.array([-lwo[i][0]-np.sqrt(PP_o[i][k][0, 0]*kappa),  # rear right
                                            -lwo[i][1]-np.sqrt(PP_o[i][k][2, 2]*kappa)]))
                cornersi[3].append(np.array([lwo[i][0]+np.sqrt(PP_o[i][k][0, 0]*kappa),  # front right
                                            -lwo[i][1]-np.sqrt(PP_o[i][k][2, 2]*kappa)]))
            cornerso.append(cornersi)
        return cornerso

    def add_velocity_cornerso(self, x0_ev, x0_tv):
        cornerso = []
        for i in range(len(self._cornerso)):
            cornersi = [[] for _ in self._cornerso[i]]
            if x0_ev[0]<x0_tv[i][0] and x0_ev[3]>x0_tv[i][1]:
                #   only if ev is behind and faster
                #   extra distance due to different speed (zero as y component)
                ddx = np.array([-(x0_ev[3]**2-x0_tv[i][1]**2)/2/self._uminev[0], 0])
            else:
                ddx = np.array([0, 0])
            if ddx[0]<0: breakpoint()
            for k in range(len(self._cornerso[i][0])):
                cornersi[0].append(self._cornerso[i][0][k]+ddx)
                cornersi[1].append(self._cornerso[i][1][k]-ddx)
                cornersi[2].append(self._cornerso[i][2][k]-ddx)
                cornersi[3].append(self._cornerso[i][3][k]+ddx)
            cornerso.append(cornersi)
        return cornerso

    def prediction_covariance(Ao, Bo, Ko, N, P_w, P_0):
        PP_tv = []
        for i in range(len(Ko)):
            Acl = Ao+Bo@Ko[i]
            PP_tvi = [P_0[i]]
            for k in range(N):
                PP_tvi.append(Bo@P_w[i]@(Bo.T)+Acl@PP_tvi[-1]@(Acl.T))
            PP_tv.append(PP_tvi[1:])
        return PP_tv

    def predict_trajectory_obstacles(self, x0_tv, xrefo):
        xx_tv = []
        for i in range(len(x0_tv)):
            xx_tvi = [x0_tv[i]]
            for k in range(self._N):
                uik = np.minimum(self._umaxo[i], np.maximum(
                    self._umino[i], self._Ko[i]@(xx_tvi[-1]-xrefo[i])))
                xx_tvi.append(self._Ao@xx_tvi[-1]+self._Bo@uik)
            xx_tv.append(xx_tvi[1:])
        return xx_tv

    # Debug function for showing the constraints
    def plot_feasible_region(self, q, x0_ev=None, x0_tv=None, xx_tv=None, text=None, comparison=False):
        """
        Plot the first and Nth q-constraint section on the street (s,d-coordinate frame)

        Parameters
        ----------
        q : dict, q-constraint
        x0_ev : array, optional, State of the Ego vehicle
        x0_tv : List, optional, List of states of target vehicle
        xx_tv : List, optional, List of the prediction of the target vehicle
        """
        if self.x0_plot is None:
            self.x0_plot = x0_ev[0]
        xmin = self.x0_plot - 50
        xmax = xmin + 200
        if x0_tv is not None:
            ymin = min(-1.75, np.matrix(x0_tv)[:, 2].min()-2)
            ymax = max(8.75,  np.matrix(x0_tv)[:, 2].max()+2)
        elif xx_tv is not None:
            ymin = min(-1.75, np.array(xx_tv)[:, :, 2].min()-2)
            ymax = max(8.75,  np.array(xx_tv)[:, :, 2].max()+2)
    

        m_per_pixel = 0.1
        plt.rcParams['figure.dpi'] = 100
        x, y = np.meshgrid(np.arange(xmin, xmax, m_per_pixel),
                           np.arange(ymin, ymax, m_per_pixel))
        region = np.ones((x.shape[0], x.shape[1], self._N)).astype(bool)
        for k in range(self._N):
            for i in range(q['s'].shape[1]):
                region[:, :, k] = region[:, :, k] & (0 >= x*q['s'][k, i]+y*q['d'][k, i]+q['t'][k, i])
        region = region.astype(int)
        # regionOverlay = np.zeros((x.shape[0],x.shape[1]))
        # for k in range(self._N):
        #     # regionOverlay = regionOverlay + (self._N-k)/self._N * region[:,:,k]
        #     regionOverlay = regionOverlay +  region[:,:,k]
        # plt.imshow(regionOverlay.max() - regionOverlay , extent=(0,200,-1.75,8.75),origin="lower", cmap="gist_heat", alpha=0.5)
        # plt.imshow(region[:,:,0] , extent=(0,200,-1.75,8.75),origin="lower", cmap="Greys", alpha=1)

        plt.imshow(region[:, :, 0], extent=(xmin, xmax, ymin, ymax), origin="lower", cmap="Greens", alpha=.2)
        plt.imshow(region[:, :, self._N-1], extent=(xmin, xmax, ymin, ymax), origin="lower", cmap="Reds", alpha=.2)
        if comparison: counter = self.counter_comparison
        else:   counter = self.counter
        if text is None:
            plt.title('Iteration ' + str(counter), fontsize=8)
        else:
            plt.title('Iteration ' + str(counter) + ', ' + text, fontsize=8)

        if isinstance(x0_ev, list):
            for k in range(len(x0_ev)):
                plt.scatter(x0_ev[k][0], x0_ev[k][1])
        elif isinstance(x0_ev, np.ndarray):
            plt.scatter(x0_ev[0], x0_ev[1])

        if x0_tv is not None:
            for k in range(len(x0_tv)):
                plt.scatter(x0_tv[k][0], x0_tv[k][2],
                            facecolor='green', edgecolor='green')

        if xx_tv is not None:
            for k in range(len(xx_tv)):
                plt.scatter(xx_tv[k][0][0], xx_tv[k][0][2],
                            facecolor='red', edgecolor='red')

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.show()

    # Debug function for showing the constraints
    def plot_feasible_region_sperat(self, q, x0_ev=None, x0_tv=None, n=0, xx_tv=None, x_pred=None, title = None, comparison=False):
        """
        Plot the first and Nth q-constraint section on the street (s,d-coordinate frame)

        Parameters
        ----------
        q : dict, q-constraint
        x0_ev : array, optional, State of the Ego vehicle
        x0_tv : List, optional, List of states of target vehicle
        xx_tv : List, optional, List of the prediction of the target vehicle
        """
        if self.x0_plot is None:
            self.x0_plot = x0_ev[0]
        xmin = self.x0_plot - 50
        xmax = xmin + 200
        if x0_tv is not None:
            ymin = min(-1.75, np.matrix(x0_tv)[:, 2].min()-2)
            ymax = max(8.75,  np.matrix(x0_tv)[:, 2].max()+2)
        elif xx_tv is not None:
            ymin = min(-1.75, np.array(xx_tv)[:, :, 2].min()-2)
            ymax = max(8.75,  np.array(xx_tv)[:, :, 2].max()+2)

        m_per_pixel = 0.1
        plt.rcParams['figure.dpi'] = 100
        x, y = np.meshgrid(np.arange(xmin, xmax, m_per_pixel),
                           np.arange(ymin, ymax, m_per_pixel))
        region = np.ones((x.shape[0], x.shape[1], q['s'].shape[1])).astype(bool)
        for i in range(q['s'].shape[1]):
            region[:, :, i] = (0 >= x*q['s'][n, i]+y*q['d'][n, i]+q['t'][n, i])

        region_acc = np.ones((x.shape[0], x.shape[1], q['s'].shape[1])).astype(bool)
        region_acc[:, :, 0] =  region[:, :, 0]
        for i in range(q['s'].shape[1]-1):
            region_acc[:, :, i+1] = region_acc[:, :, i] & region[:, :, i+1]

        region = region.astype(int)
        region_acc = region_acc.astype(int)
        
        if xx_tv is not None:
            xx_tv = np.array(xx_tv)

        fig, axs = plt.subplots(q['s'].shape[1], 1, figsize=(20, 12))
        if isinstance(title, str):
            axs[0].title.set_text(title)
            
        for i in range(q['s'].shape[1]):
            axs[i].imshow(region[:, :, i], extent=(xmin, xmax, ymin, ymax), origin="lower", cmap="Greens", alpha=.3)
            axs[i].imshow(region_acc[:, :, i], extent=(xmin, xmax, ymin, ymax), origin="lower", cmap="Greys", alpha=.1)
            if isinstance(x0_ev, list):
                for k in range(len(x0_ev)):
                    axs[i].scatter(x0_ev[k][0], x0_ev[k][1])
            elif isinstance(x0_ev, np.ndarray):
                axs[i].scatter(x0_ev[0], x0_ev[1])

            if x0_tv is not None:
                for k in range(len(x0_tv)):
                    axs[i].scatter(x0_tv[k][0], x0_tv[k][2], facecolor='green', edgecolor='green')

            if x0_tv is not None:
                axs[i].scatter(x0_tv[i][0], x0_tv[i][2], facecolor='red', edgecolor='red')
                
            if xx_tv is not None:
                for k in range(len(xx_tv)):
                    axs[i].scatter(xx_tv[k,:,0], xx_tv[k,:,2], s=1, facecolor='darkgreen', edgecolor='green')

            if xx_tv is not None:
                axs[i].scatter(xx_tv[i,:,0], xx_tv[i,:,2], s=1, facecolor='red', edgecolor='red')

            if x_pred is not None:
                axs[i].scatter(x_pred[0, :], x_pred[1, :], s=1, facecolor='blue', edgecolor='blue')

            axs[i].set_ylabel('TV' + str(i) + '\n' + q['cases'][i])
            axs[i].set_xlim(xmin, xmax)
            axs[i].set_ylim(ymin, ymax)
            if comparison: counter = self.counter_comparison
            else:   counter = self.counter
        axs[-1].set_xlabel('Iteration ' + str(counter) + ', Prediction ' + str(n) +
                           ', Blue: EV, Red: TV which produces the shown constraint, Green: all other TVs, Green area: Feasible region due to red TV', fontsize=8)
        plt.show()

    def get_reachable_sets(self, x0_tv, xrefo, x0_ev):
        corners = [dict() for x0 in x0_tv]
        convex_hull = [[[] for k in range(4)] for x0 in x0_tv]
        cornersoexp = [[[] for k in range(4)] for x0 in x0_tv]
        cornersovar = [[[] for k in range(4)] for x0 in x0_tv]
        for i in range(len(x0_tv)):
            lwoi = self._lwo[i]
            #   no braking distance here, worst-case scenario approach
            #   initialize with min/max x0tv and with physical shape (full length, to account for ev!)
            corners[i]['fl'] = [np.array([x0_tv[i][0]+self._v_amp[i][0]+lwoi[0],  # max posx
                                          x0_tv[i][1] + \
                                          self._v_amp[i][1],  # max vx
                                          x0_tv[i][2]+self._v_amp[i][2] + \
                                          lwoi[1],  # max posy
                                          x0_tv[i][3]+self._v_amp[i][3]])]  # max vy
            corners[i]['rr'] = [np.array([x0_tv[i][0]-self._v_amp[i][0]-lwoi[0],  # min posx
                                          x0_tv[i][1] - \
                                          self._v_amp[i][1],  # min vx
                                          x0_tv[i][2]-self._v_amp[i][2] - \
                                          lwoi[1],  # min posy
                                          x0_tv[i][3]-self._v_amp[i][3]])]  # min vy
            corners[i]['cen'] = [x0_tv[i]]  # expected value
            for k in range(self._N):
                corners[i]['fl'].append(self._Ao@corners[i]['fl'][-1]+self._Bo@self._umaxo[i])  # max ax, max ay
                corners[i]['rr'].append(self._Ao@corners[i]['rr'][-1]+self._Bo@self._umino[i])  # min ax, min ay
                #   take convex hull (NB: max/min is needed because the order is swapped if a TV goes in other direction)
                convex_hull[i][0].append(np.array([max(corners[i]['fl'][k][0], corners[i]['fl'][k+1][0]),  # max x
                                                   max(corners[i]['fl'][k][2], corners[i]['fl'][k+1][2])]))  # max y
                convex_hull[i][1].append(np.array([min(corners[i]['rr'][k][0], corners[i]['rr'][k+1][0]),  # min x
                                                   max(corners[i]['fl'][k][2], corners[i]['fl'][k+1][2])]))  # max y
                convex_hull[i][2].append(np.array([min(corners[i]['rr'][k][0], corners[i]['rr'][k+1][0]),  # min x
                                                   min(corners[i]['rr'][k][2], corners[i]['rr'][k+1][2])]))  # min y
                convex_hull[i][3].append(np.array([max(corners[i]['fl'][k][0], corners[i]['fl'][k+1][0]),  # max x
                                                   min(corners[i]['rr'][k][2], corners[i]['rr'][k+1][2])]))  # min y
                corners[i]['cen'].append(self._Ao@corners[i]['cen'][-1]+self._Bo@np.minimum(self._umaxo[i],
                                                                                            np.maximum(self._umino[i], self._Ko[i]@(corners[i]['cen'][-1]-xrefo[i]))))
                #   truncated Gaussian corners
                mu, sigma = truncated_gaussian(corners[i]['cen'][-1][0:4:2]+lwoi,    # max x, max y
                                               np.diag(self._PP_o[i][k])[0:4:2],
                                               np.minimum(corners[i]['rr'][k][0:4:2],
                                               corners[i]['rr'][k+1][0:4:2])+lwoi,  # max x, max y
                                               np.maximum(corners[i]['fl'][k][0:4:2],
                                                          corners[i]['fl'][k+1][0:4:2]))    # max x, max y
                cornersoexp[i][0].append(mu)
                cornersovar[i][0].append(sigma)
                mu, sigma = truncated_gaussian(corners[i]['cen'][-1][0:4:2]
                                               + np.multiply(lwoi, np.array([-1, 1])),    # min x, max y
                                               np.diag(self._PP_o[i][k])[0:4:2],
                                               np.minimum(corners[i]['rr'][k][0:4:2],   # min x, max y
                                                          corners[i]['rr'][k+1][0:4:2])
                                               + np.multiply(lwoi, np.array([0, 1])),
                                               np.maximum(corners[i]['fl'][k][0:4:2],  # min x, max y
                                                          corners[i]['fl'][k+1][0:4:2])
                                               + np.multiply(lwoi, np.array([-1, 0])))
                cornersoexp[i][1].append(mu)
                cornersovar[i][1].append(sigma)
                mu, sigma = truncated_gaussian(corners[i]['cen'][-1][0:4:2]-lwoi,   # min x, min y
                                               np.diag(self._PP_o[i][k])[0:4:2],
                                               np.minimum(corners[i]['rr'][k][0:4:2],  # min x, min y
                                                          corners[i]['rr'][k+1][0:4:2]),
                                               np.maximum(corners[i]['fl'][k][0:4:2],
                                                          corners[i]['fl'][k+1][0:4:2])-lwoi)  # min x, min y
                cornersoexp[i][2].append(mu)
                cornersovar[i][2].append(sigma)
                mu, sigma = truncated_gaussian(corners[i]['cen'][-1][0:4:2]
                                               + np.multiply(lwoi, np.array([1, -1])),    # max x, min y
                                               np.diag(self._PP_o[i][k])[0:4:2],
                                               np.minimum(corners[i]['rr'][k][0:4:2],   # max x, min y
                                                          corners[i]['rr'][k+1][0:4:2])
                                               + np.multiply(lwoi, np.array([1, 0])),
                                               np.maximum(corners[i]['fl'][k][0:4:2],  # max x, min y
                                                          corners[i]['fl'][k+1][0:4:2])
                                               + np.multiply(lwoi, np.array([0, -1])))
                cornersoexp[i][3].append(mu)
                cornersovar[i][3].append(sigma)
        return convex_hull, cornersoexp, cornersovar

    def print_applied_method(self, comparison=False):
        if comparison:
            texts = self.applied_method_comparison
            print('\nComparison (SMPC+FTP)')
        else:
            texts = self.applied_method
            print('\nSuggested method (SMPC+CVPM)')
        for j in range(len(texts)):
            print('Iteration: ', j, texts[j])
        print()
        
    
    def print_average_stage_cost(self):
        self.__cost = np.average(self.stage_cost_sequence)
        print('\n\nAverage stage cost: ', self.__cost)
        self.__cost_comparison = np.average(self.stage_cost_sequence_comparison)
        print('Average stage cost (comparison): ', self.__cost_comparison)
        
        
    def add_iteration_time_counters(self, comparison=False):
        if comparison:
            self._comp_time_smpc_comparison.append(-1)
            self._comp_time_ftp_now_comparison.append(-1)
            self._comp_time_ftp_next_comparison.append(-1)
            # self._comp_time_smpc_branch_comparison.append(-1)
            # self._comp_time_ftp_branch_comparison.append(-1)
        else:
            self._comp_time_smpc.append(-1)
            self._comp_time_cvpm1.append(-1)
            self._comp_time_cvpm2.append(-1)
            self._comp_time_isFeasible.append(-1)
            self._comp_time_case_cvpm.append(-1)
            
    def print_computation_times(self):
        self.compute_comp_time_branches()
        
        print('\nSuggested method (SMPC+CVPM)')
        
        self.__time_smpc_average = np.average(self.where_used(self._comp_time_smpc))
        self.__time_smpc_max = np.max(self.where_used(self._comp_time_smpc))
        print('Average computation time (SMPC): ', self.__time_smpc_average)
        print('Maximum computation time (SMPC): ', self.__time_smpc_max)
        
        if self.where_used(self._comp_time_cvpm1) != []:
            self.__time_cvpm1_average = np.average(self.where_used(self._comp_time_cvpm1))
            self.__time_cvpm1_max = np.max(self.where_used(self._comp_time_cvpm1))
            print('Average computation time (CVPM c1): ', self.__time_cvpm1_average)
            print('Maximum computation time (CVPM c1): ', self.__time_cvpm1_max)
        else:
            self.__time_cvpm1_average = -1
            self.__time_cvpm1_max = -1
            
        if self.where_used(self._comp_time_cvpm2) != []:
            self.__time_cvpm2_average = np.average(self.where_used(self._comp_time_cvpm2))
            self.__time_cvpm2_max = np.max(self.where_used(self._comp_time_cvpm2))
            print('Average computation time (CVPM c2): ', self.__time_cvpm2_average)
            print('Maximum computation time (CVPM c2): ', self.__time_cvpm2_max)
        else:
            self.__time_cvpm2_average = -1
            self.__time_cvpm2_max = -1
            
        self.__time_isfeasible_average = np.average(self.where_used(self._comp_time_isFeasible))
        self.__time_isfeasible_max = np.max(self.where_used(self._comp_time_isFeasible))
        print('Average computation time (isFeasible): ', self.__time_isfeasible_average)
        print('Maximum computation time (isFeasible): ', self.__time_isfeasible_max)
        
        if self.where_used(self._comp_time_case_cvpm) != []:
            self.__time_cvpmcase_average = np.average(self.where_used(self._comp_time_case_cvpm))
            self.__time_cvpmcase_max = np.max(self.where_used(self._comp_time_case_cvpm))
            print('Average computation time (Check case CVPM): ', self.__time_cvpmcase_average)
            print('Maximum computation time (Check case CVPM): ', self.__time_cvpmcase_max)
        else:
            self.__time_cvpmcase_average = -1
            self.__time_cvpmcase_max = -1
            
        self.__time_smpcbranch_average = np.average(self.comp_time_smpc_branch)
        self.__time_smpcbranch_max = np.max(self.comp_time_smpc_branch)
        print('Average computation time (SMPC branch): ', self.__time_smpcbranch_average)
        print('Maximum computation time (SMPC branch): ', self.__time_smpcbranch_max)
        
        if self.where_used(self.comp_time_cvpm_branch) != []:
            self.__time_cvpmbranch_average = np.average(self.where_used(self.comp_time_cvpm_branch))
            self.__time_cvpmbranch_max = np.max(self.where_used(self.comp_time_cvpm_branch))
            print('Average computation time (CVPM branch): ', self.__time_cvpmbranch_average)
            print('Maximum computation time (CVPM branch): ', self.__time_cvpmbranch_max)
        else:
            self.__time_cvpmbranch_average = -1
            self.__time_cvpmbranch_max = -1
        
            
        print('\nComparison (SMPC+FTP)')
        
        self.__time_smpc_comparison_average = np.average(self.where_used(self._comp_time_smpc_comparison))
        self.__time_smpc_comparison_max = np.max(self.where_used(self._comp_time_smpc_comparison))
        print('Average computation time (SMPC): ', self.__time_smpc_comparison_average)
        print('Maximum computation time (SMPC): ', self.__time_smpc_comparison_max)
        
        if self.where_used(self._comp_time_ftp_now_comparison) != []:
            self.__time_ftpnow_comparison_average = np.average(self.where_used(self._comp_time_ftp_now_comparison))
            self.__time_ftpnow_comparison_max = np.max(self.where_used(self._comp_time_ftp_now_comparison))
            print('Average computation time (FTP now): ', self.__time_ftpnow_comparison_average)
            print('Maximum computation time (FTP now): ', self.__time_ftpnow_comparison_max)
        else:
            self.__time_ftpnow_comparison_average = -1
            self.__time_ftpnow_comparison_max = -1
            
        if self.where_used(self._comp_time_ftp_next_comparison) != []:
            self.__time_ftpnext_comparison_average = np.average(self.where_used(self._comp_time_ftp_next_comparison))
            self.__time_ftpnext_comparison_max = np.max(self.where_used(self._comp_time_ftp_next_comparison))
            print('Average computation time (FTP next): ', self.__time_ftpnext_comparison_average)
            print('Maximum computation time (FTP next): ', self.__time_ftpnext_comparison_max)
        else:
            self.__time_ftpnext_comparison_average = -1
            self.__time_ftpnext_comparison_max = -1
            
        self.__time_smpcbranch_comparison_average = np.average(self.comp_time_smpc_branch_comparison)
        self.__time_smpcbranch_comparison_max = np.max(self.comp_time_smpc_branch_comparison)
        print('Average computation time (SMPC branch comparison): ', self.__time_smpcbranch_comparison_average)
        print('Maximum computation time (SMPC branch comparison): ', self.__time_smpcbranch_comparison_max)
        
        if self.where_used(self.comp_time_ftp_branch_comparison) != []:
            self.__time_ftpbranch_comparison_average = np.average(self.where_used(self.comp_time_ftp_branch_comparison))
            self.__time_ftpbranch_comparison_max = np.max(self.where_used(self.comp_time_ftp_branch_comparison))
            print('Average computation time (FTP branch comparison): ', self.__time_ftpbranch_comparison_average)
            print('Maximum computation time (FTP branch comparison): ', self.__time_ftpbranch_comparison_max)
        else:
            self.__time_ftpbranch_comparison_average = -1
            self.__time_ftpbranch_comparison_max = -1
            
            
            
    def where_used(self, list_):
        return [el for el in list_ if el>0]
    
    def compute_comp_time_branches(self):
        
        self.comp_time_smpc_branch = []
        self.comp_time_cvpm_branch = []
        self.comp_time_smpc_branch_comparison = []
        self.comp_time_ftp_branch_comparison = self.where_used(self._comp_time_ftp_now_comparison)
        for k in range(len(self._comp_time_smpc)):
            #   smpc branch
            curr_smpc = self._comp_time_smpc[k]
            if self._comp_time_isFeasible[k]!=-1:
                curr_smpc+=self._comp_time_isFeasible[k]
            self.comp_time_smpc_branch.append(curr_smpc)
            
            #   cvpm branch
            if self._comp_time_case_cvpm[k]!=-1:
                curr_cvpm = self._comp_time_case_cvpm[k]
                if self._comp_time_cvpm1[k]!=-1 and self._comp_time_cvpm2[k]!=-1:
                    print('Error, apparently both cvpm problem solved, although not')
                    breakpoint()
                else:
                    if self._comp_time_cvpm2[k]!=-1:
                        curr_cvpm+=self._comp_time_cvpm2[k]
                self.comp_time_cvpm_branch.append(curr_cvpm)
            
            #   smpc branch comparison
            curr_smpc_comparison = self._comp_time_smpc_comparison[k]
            if self._comp_time_ftp_next_comparison[k]!=-1:
                curr_smpc_comparison+=self._comp_time_ftp_next_comparison[k]
            self.comp_time_smpc_branch_comparison.append(curr_smpc_comparison)
        
