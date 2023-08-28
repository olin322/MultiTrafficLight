import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gymnasium.envs.registration import register

from Game import Game
from Vehicle import Vehicle
from rewards import RewardMap
from TrafficLight import TrafficLight

from overrides import override




class SingleTrafficLightEnv(gym.Env, Game):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, ):
        delta_t=0.1
        super().__init__(delta_t)
        totalTrafficLights=1       
        ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, delta_t)
        trafficLight_1  = TrafficLight("1",  100,  "green", 10, delta_t)
        trafficLights = [
                    trafficLight_1
                    ]
        self.actors.append(ego_vehicle)
        self.actors.append(trafficLight_1)
        rewardMap = RewardMap(ego_vehicle, trafficLights)
        self.totalTrafficLights = totalTrafficLights
        self.rewardMap = rewardMap
        self.num_envs = 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        num_obs = self.totalTrafficLights * 2 + 3
        self.observation_space = spaces.Box(low=0, high=10000, shape=(num_obs,), dtype=np.float32)
       
    def step(self, action) -> GymStepReturn: 
        self.rl_tick(action)
        observation = self._get_observation()
        terminated = self.rewardMap.tick(action)
        reward = self.rewardMap.getStepReward() # update reward and reward map
        terminated = terminated \
                    | bool(self.actors[self._get_ego_vehicle_index()].getLocation() >= 200.0) \
                    | self.frame > 2500
        truncated = False # unnecessary to truncate anything
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        for actor in self.actors:
            actor.reset()
        self.rewardMap.reset()
        observation = self._get_observation()
        if (seed):
            print(seed)
        info = {}
        return observation, info

    def render(self):
        return None

    def close(self):
        pass

    def _get_observation(self) -> spaces.Box:
        ego_vehicle = self.actors[self._get_ego_vehicle_index()]
        obs = []
        obs.append(ego_vehicle.getLocation())
        obs.append(ego_vehicle.getSpeed())
        if (self._find_next_light()):
            dis = self._find_next_light().getLocation() - ego_vehicle.getLocation()
            obs.append(dis)
        else:
            obs.append(-1)
        for a in self.actors:
            if(self._find_Actor_Type(a) == "TrafficLight"):
                if(a.getLocation() < ego_vehicle.getLocation()):
                    obs.append(-1)
                    obs.append(-1)
                else:
                    obs.append(a.getCountdown())
                    obs.append(a.getPhaseInFloat())
        observation = np.array(obs, dtype=np.float32)
        return observation


register(
    # unique identifier for the env `name-version`
    id = "SingleTrafficLight-v1",
    # path to the class for creating the env
    # entry_point also accept a class as input (and not only a string)
    entry_point = "envs.SimpleEnvs:SingleTrafficLightEnv",
    # max_episode_steps is not necessary in this env as ev 
    # will always arrive at destination
    max_episode_steps = 1e4
)


class SingleTrafficLightEnvMultiProc(gym.Env, Game):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, 
                totalTrafficLights=1, 
                delta_t=0.1, 
                ):
        super().__init__(delta_t)
        ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, delta_t, speed=0)
        trafficLight_1 = TrafficLight("1", 100, "green", 10, delta_t)
        self.actors.append(ego_vehicle)
        self.actors.append(trafficLight_1)
        trafficLights = [trafficLight_1]
        rewardMap = RewardMap(delta_t, ego_vehicle, trafficLights)
        self.totalTrafficLights = len(trafficLights)
        self.rewardMap = rewardMap
        self.num_envs = 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        num_obs = self.totalTrafficLights * 2 + 3
        self.observation_space = spaces.Box(low=0, high=10000, shape=(num_obs,), dtype=np.float32)
       
    def step(self, action) -> GymStepReturn: 
        self.rl_tick(action)
        observation = self._get_observation()
        terminated = self.rewardMap.tick(action)
        reward = self.rewardMap.getStepReward() # update reward and reward map
        terminated = terminated \
                     | bool(self.actors[self._get_ego_vehicle_index()].getLocation() >= 299.9) \
                     | bool(self.frame > 5000)
        truncated = False # unnecessary to truncate anything
        info = {}
        return observation, reward, terminated, truncated, info

    @override
    def reset(self, seed=None, options=None):
        for actor in self.actors:
            actor.reset()
        self.rewardMap.reset()
        self.frame = 0
        self.simulation_time = 0
        observation = self._get_observation()
        if (seed):
            print(seed)
        info = {}
        return observation, info

    def render(self):
        return None

    def close(self):
        pass

    def _get_observation(self) -> spaces.Box:
        # OBS SPACE:
        # ego_vehicle location, ego_vehicle speed, ev_distance_to_nextLight
        # trafficLight1_CountDown, trafficLight2_PhaseInFloat
        ego_vehicle = self.actors[self._get_ego_vehicle_index()]
        obs = []
        obs.append(ego_vehicle.getLocation())
        obs.append(ego_vehicle.getSpeed())
        if (self._find_next_light()):
            dis = self._find_next_light().getLocation() - ego_vehicle.getLocation()
            obs.append(dis)
        else:
            obs.append(-1)
        for a in self.actors:
            if(self._find_Actor_Type(a) == "TrafficLight"):
                if(a.getLocation() < ego_vehicle.getLocation()):
                    obs.append(-1)
                    obs.append(-1)
                else:
                    obs.append(a.getCountdown())
                    obs.append(a.getPhaseInFloat())
        observation = np.array(obs, dtype=np.float32)
        return observation


register(
    # unique identifier for the env `name-version`
    id = "SingleTrafficLightMultiProc-v1",
    # path to the class for creating the env
    # entry_point also accept a class as input (and not only a string)
    entry_point = "envs.SimpleEnvs:SingleTrafficLightEnvMultiProc",
    # max_episode_steps is not necessary in this env as ev 
    # will always arrive at destination
    max_episode_steps = 4e3
)



class TwoTrafficLightEnvMultiProc(gym.Env, Game):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, 
                totalTrafficLights=2, 
                delta_t=0.1, 
                ):
        super().__init__(delta_t)
        ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, delta_t, speed=0)
        trafficLight_1 = TrafficLight("1", 100, "green", 10, delta_t)
        trafficLight_2 = TrafficLight("2", 200, "red",   17, delta_t)
        self.actors.append(ego_vehicle)
        self.actors.append(trafficLight_1)
        self.actors.append(trafficLight_2)
        trafficLights = [trafficLight_1, trafficLight_2]
        rewardMap = RewardMap(delta_t, ego_vehicle, trafficLights)
        self.totalTrafficLights = len(trafficLights)
        self.rewardMap = rewardMap
        self.num_envs = 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        num_obs = self.totalTrafficLights * 2 + 3
        self.observation_space = spaces.Box(low=0, high=10000, shape=(num_obs,), dtype=np.float32)
       
    def step(self, action) -> GymStepReturn: 
        self.rl_tick(action)
        observation = self._get_observation()
        terminated = self.rewardMap.tick(action)
        reward = self.rewardMap.getStepReward() # update reward and reward map
        terminated = terminated \
                     | bool(self.actors[self._get_ego_vehicle_index()].getLocation() >= 299.9) \
                     | bool(self.frame > 5000)
        truncated = False # unnecessary to truncate anything
        info = {}
        return observation, reward, terminated, truncated, info

    @override
    def reset(self, seed=None, options=None):
        for actor in self.actors:
            actor.reset()
        self.rewardMap.reset()
        self.frame = 0
        self.simulation_time = 0
        observation = self._get_observation()
        if (seed):
            print(seed)
        info = {}
        return observation, info

    def render(self):
        return None

    def close(self):
        pass

    def _get_observation(self) -> spaces.Box:
        ego_vehicle = self.actors[self._get_ego_vehicle_index()]
        obs = []
        obs.append(ego_vehicle.getLocation())
        obs.append(ego_vehicle.getSpeed())
        if (self._find_next_light()):
            dis = self._find_next_light().getLocation() - ego_vehicle.getLocation()
            obs.append(dis)
        else:
            obs.append(-1)
        for a in self.actors:
            if(self._find_Actor_Type(a) == "TrafficLight"):
                if(a.getLocation() < ego_vehicle.getLocation()):
                    obs.append(-1)
                    obs.append(-1)
                else:
                    obs.append(a.getCountdown())
                    obs.append(a.getPhaseInFloat())
        observation = np.array(obs, dtype=np.float32)
        return observation


register(
    # unique identifier for the env `name-version`
    id = "TwoTrafficLightMultiProc-v1",
    # path to the class for creating the env
    # entry_point also accept a class as input (and not only a string)
    entry_point = "envs.SimpleEnvs:TwoTrafficLightEnvMultiProc",
    # max_episode_steps is not necessary in this env as ev 
    # will always arrive at destination
    max_episode_steps = 5e3
)