import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.observation import Observation
from bayesfilter.model import StateTransitionModel
from bayesfilter.smoothing import RTS
    

def constant_transition(x, delta_t_s):
    return np.array([x[0], x[1]])


def observe_speed(x):
    return np.array([x[0]])


def observe_yaw_rate(x):
    return np.array([x[0]*x[1]])


def post_process_speed_and_yaw(speed_list_mps, yaw_list_rads_per_sec, time_list_s):
    transition_model = StateTransitionModel(
        constant_transition, 
        np.diag([1e-1, 1e-4])
    )
    initial_state = Gaussian(np.array([0.0, 0.0]), np.diag([5.0, 0.5]))
    print(initial_state)
    filter = BayesianFilter(transition_model, initial_state)

    speed_obs_noise_mps = 0.5
    yaw_obs_noise_rads_per_sec = np.deg2rad(20.0)
    observations = []
    obs_times_s = []
    for s, y, t in zip(speed_list_mps, yaw_list_rads_per_sec, time_list_s):
        observations.append(
            Observation(
                s*np.eye(1), 
                speed_obs_noise_mps*np.eye(1), 
                observe_speed
            )
        )
        obs_times_s.append(t)
        observations.append(
            Observation(
                y*np.eye(1), 
                yaw_obs_noise_rads_per_sec*np.eye(1), 
                observe_yaw_rate
            )
        )
        obs_times_s.append(t)
    
    filter_states, filter_times = filter.run(observations, obs_times_s, 100.0)
    smoother = RTS(filter)
    smoother_states = smoother.apply(filter_states, filter_times)
    output_speeds_mps = np.array([s.mean()[0] for s in smoother_states])
    output_curvatures_invm = np.array([s.mean()[1] for s in smoother_states])
    return output_speeds_mps, output_curvatures_invm, filter_times
