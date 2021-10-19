import os
import argparse
import grid2op
import gym
import numpy as np
import time
import random
from grid2op.PlotGrid import PlotMatplot

class MaxTimestepWrapper(gym.Wrapper):
    def __init__(self, env):        
        gym.Wrapper.__init__(self, env)
        self.timestep = 0

    def step(self, action, **kwargs):
        self.timestep += 1
        obs, reward, done, info = self.env.step(action, **kwargs)
        if self.timestep >= MAX_TIMESTEP:
            done = True
            info["timeout"] = True
        else:
            info["timeout"] = False
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.timestep = 0
        return self.env.reset(**kwargs)

class ObsTransformerWrapper(gym.Wrapper):
    def __init__(self, env):        
        gym.Wrapper.__init__(self, env)
        
        self.has_overflow = False # Add an attribute to mark whether the env has overflowed lines..     

    def step(self, action, **kwargs):
        raw_obs, reward, done, info = self.env.step(action, **kwargs)
        self.has_overflow = self._has_overflow(raw_obs)        
        return raw_obs, reward, done, info

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self.has_overflow = self._has_overflow(raw_obs)        
        return raw_obs
    
    def _has_overflow(self, obs):
        has_overflow = False
        if obs is not None and not any(np.isnan(obs.rho)):
            has_overflow = any(obs.rho > 1.0)
        return has_overflow

class ActionMappingWrapper(gym.Wrapper):
    def __init__(self, env):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """
        gym.Wrapper.__init__(self, env)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        self.low_bound = self.env.action_space.low
        self.high_bound = self.env.action_space.high
        assert np.all(self.high_bound >= self.low_bound)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, model_output_act):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """
        assert np.all(((model_output_act<=1.0 + 1e-3), (model_output_act>=-1.0 - 1e-3))), \
            'the action should be in range [-1.0, 1.0]'
        mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
            (self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)
        return self.env.step(mapped_action)

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):        
        gym.Wrapper.__init__(self, env)

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)

        shaping_reward = 2.0 - obs.rho.max()
        if obs.rho.max() < 1.0:
            shaping_reward += 1.0

        if done and not info["timeout"]:
            shaping_reward = -10.0

        info["origin_reward"] = reward

        return obs, shaping_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def get_env():
    #dataset path
    dataset = "/home/bml/GridSolution/data_grid2op/l2rpn_neurips_2020_track1_small"
    env = grid2op.make_redispatch_env(dataset=dataset, max_timestep=MAX_TIMESTEP)    
    # NOTE: The order of the following wrappers matters.
    env = MaxTimestepWrapper(env)
    env = RewardShapingWrapper(env)
    env = ObsTransformerWrapper(env)
    env = ActionMappingWrapper(env)
    return env

def has_overflow(raw_obs):
    if raw_obs is not None and not any(np.isnan(raw_obs.rho)):
        has_overflow = any(raw_obs.rho > 1.0)
    return has_overflow   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_id", default=0, type=int)    # id of subset of the same city
    args = parser.parse_args()

    MAX_TIMESTEP = 288 * 7
    MAX_EPISODE = 1000
    
    episode_index = 0    
    env = get_env()
    do_nothing_action = np.zeros(env.action_space.shape)    

    while episode_index < MAX_EPISODE:  
        step = 0      
        raw_obs = env.reset()               

        # case1: if has overflow in step 0, continue        
        if has_overflow(raw_obs):            
            continue
        obs_as_vec = raw_obs.to_vect() 
        obs_as_vec = np.append(obs_as_vec, episode_index)  # obs_as_vec end with episode_index    

        while True:    
            step+=1
            next_obs, reward, done, info = env.step(do_nothing_action) 
            next_obs_as_vec = next_obs.to_vect() 
            next_obs_as_vec = np.append(next_obs_as_vec, episode_index)
            obs_as_vec = np.vstack((obs_as_vec, next_obs_as_vec))         
            if done:                
                break  
            if has_overflow(next_obs):
                break 

        # case2: more than MAX_TIMESTEP, continue     
        if done:
            continue

        # case3: steps less than 100, continue
        if step<110:            
            continue 
        
        if episode_index==0:
            city_vect = obs_as_vec  
        else:
            city_vect = np.vstack((city_vect, obs_as_vec))
        # print('episode index', episode_index)
        episode_index+=1  

    subset_name = 'city1' + '_subset' + str(args.subset_id)
    np.save(subset_name, city_vect)   
    
