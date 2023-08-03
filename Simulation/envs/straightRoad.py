import gymnasium as gym
import numpy as np
from gymnasium import spaces
from World import World
from rewards import RewardMap

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gymnasium.envs.registration import register

from overrides import override


class StraightRoadEnv(gym.Env, World):
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
                totalTrafficLights: int, 
                delta_t: float, 
                rewardMap: RewardMap):
        # super().__init__() gym.env has no constructor
        super().__init__(delta_t)
        self.totalTrafficLights = totalTrafficLights
        self.rewardMap = rewardMap
        self.num_envs = 1

        """
        @action_space is defined as acceleration or deacceleration of ego_vehicle
        the value action_space can take is in [max_deacceleration, max_acceleration]
        which are defined as parameters of constructor of Vehicle class
        max_deacceleration and max_acceleration are currently set to 2 and 2, respectively
        """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


        ###############################################################################################
        # THIS PART IS NOT USED AS SAC ALGORITHM REQUIRES CONTINUES-VALUE ACTION SPACE
        # """
        # | Num |                    Action                      | Control Min | Control Max | Unit |
        # |  0  | accelerate at maximum acceleration             |     N/A     |     N/A     |      |
        # |  1  | deaccelerate at maximum deacceleration         |     N/A     |     N/A     |      |
        # |  2  | move at current speed                          |     N/A     |     N/A     |      |
        # """
        # self.action_space = spaces.Discrete(3)
        ###############################################################################################

        """
        The first element in Tuple of the observation_space
        | Num |                 Observation                    | Value Min |   Vale Max   | Unit  |
        |  0  | location of ego_vehicle at previous step       |     0     |     10000    |   m   |
        |  1  | speed/velocity of ego_vehicle at previous step |     0     |     16.67    |  m/s  |
        |  2  | number of traffic light ahead                  |     0     |              |       |
        |     |                                                |           |              |       |

        """

        """
        The second element in Tuple is a matrix of shape (totalTrafficLights, 3)
        representing SPaT of all traffic lights, the matrix looks like follows
        [
            [location of light 0, countDown of light 0, phase of light 0],
            [location of light 1, countDown of light 1, phase of light 1],
                                            .
                                            .
                                            .
            [location of light n-1, countDown of light n-1, phase of light n-1]
        ]
        
        Note that when `shape=(n, )`, it represents a 1 x n array, 
        or a 1-D array with n elements;
        when `shape=(i, j, )`, it represents a i x j matrix, 
        or 2-D array with i rows and j cols,
        and the low high values are inclusive.
        """
        # num_obs = self.totalTrafficLights * 3 + 2
        # self.observation_space = spaces.Box(low=0, high=10000, shape=(num_obs,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "ego_vehicle_location": spaces.Box(low=-0.1, high=10000.0, shape=(1,), dtype=np.float64),
                "ego_vehicle_speed": spaces.Box(low=-0.1, high=16.67, shape=(1,), dtype=np.float64),
                # "num_of_traffic_lights_ahead": spaces.Discrete(totalTrafficLights),
                "traffic_lights_states": spaces.Box(
                    low=-1,
                    high=10000,
                    shape=(totalTrafficLights*3, ),
                    dtype=np.float64
                )
                # "traffic_lights_states": spaces.Box(
                #     low=np.array([[0, 0, 0] * totalTrafficLights]).reshape(totalTrafficLights, 3), 
                #     high=np.array([[10000.0, 300, 2]] * totalTrafficLights).reshape(totalTrafficLights, 3),
                #     shape=(totalTrafficLights, 3,), 
                #     dtype=np.float64
                # )
            }
        )
        
    """
        GymEnv = Union[gym.Env, vec_env.VecEnv]
        GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
        GymResetReturn = Tuple[GymObs, Dict]
        GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
        in this case the return value would be a tuple defined as follows
        tuple(spaces.Tuple, float, bool, bool, int):
        @reward is the reward gained from one step
    """
    # update actions first, then get_obs
    def step(self, action) -> GymStepReturn: 
        self.rl_tick(action)
        observation = self._get_observation()
        self.rewardMap.tick()
        reward = self.rewardMap.getStepReward() # update reward and reward map
        # unnecessary to check max_step as ev will always arrive destination
        terminated = bool(self.actors[self._get_ego_vehicle_index()].getLocation() >= 10000.0)
        truncated = False # unnecessary to truncate anything
        info = {}
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None) -> tuple:
        # -> tuple(spaces.Tuple(spaces.Box, spaces.Box, spaces.Discrete, spaces.Box), info):
        # should re-initialize all traffic light status 
        # reset all rewards ?
        # and place the ego_vehicle to correct location ?

        # return value `info` is not currently used, set to None for now
        for actor in self.actors:
            actor.reset()
        self.rewardMap.reset()
        observation = self._get_observation()
        if (seed):
            print(seed)
        info = {}
        return observation, info

    def render(self):
        # Return type Optional[ndarray]
        return None

    def close(self):
        pass

    def _get_observation(self) -> spaces.Dict:
        # need a pointer in World class that points to ego_vehicle
        ego_vehicle = self.actors[self._get_ego_vehicle_index()]
        # obs = []
        # obs.append(ego_vehicle.getLocation())
        # obs.append(ego_vehicle.getSpeed())
        obs = []
        for a in self.actors:
            if(self._find_Actor_Type(a) == "TrafficLight"):
                obs.append(a.getLocation())
                obs.append(a.getCountdown())
                obs.append(a.getPhaseInFloat())

        observation = dict(
            {
                "ego_vehicle_location": np.array([ego_vehicle.getLocation(), ]),
                "ego_vehicle_speed": np.array([ego_vehicle.getSpeed()], ).reshape(1,),
                # "num_of_traffic_lights_ahead": self.numTrafficLightAhead(ego_vehicle),
                "traffic_lights_states": np.array(obs, dtype=np.float64)#.reshape(self.totalTrafficLights, 3)
            }
        )
        # observation = np.array(obs, dtype=np.float32)
        return observation


register(
    # unique identifier for the env `name-version`
    id = "StraightRoad-v1",
    # path to the class for creating the env
    # entry_point also accept a class as input (and not only a string)
    entry_point = "envs.straightRoad:StraightRoadEnv",
    # max_episode_steps is not necessary in this env as ev 
    # will always arrive at destination
    max_episode_steps = 1e6
)