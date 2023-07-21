import gymnasium as gym
import numpy as np
from gymnasium import spaces
from World import World


from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from overrides import override


class StraightRoadEnv(gym.Env, World):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": ["human"], 
        "render_fps": 30
    }

    def __init__(self, 
                totalTrafficLights: int, 
                delta_t: float, 
                rewardMap: RewardMap):
        super().__init__()
        super(gym.Env, self).__init__(delta_t)
        self.totalTrafficLights = totalTrafficLights
        self.rewardMap = rewardMap

        
        """
        | Num |                    Action                      | Control Min | Control Max | Unit |
        |  0  | accelerate at maximum acceleration             |     N/A     |     N/A     |      |
        |  1  | deaccelerate at maximum deacceleration         |     N/A     |     N/A     |      |
        |  2  | move at current speed                          |     N/A     |     N/A     |      |
        """
        self.action_space = spaces.Discrete(3)


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
        self.observation_space = 
            spaces.Tuple(
                spaces.Box(low=0.0, high=10000.0, shape=(1,), dtype=np.float32),
                spaces.Box(low=0.0, high=16.67, shape=(1,), dtype=np.float32)
                spaces.Discrete(totalTrafficLights)
                spaces.Box(
                    low=np.array([[0.0, 0, 0] * totalTrafficLights]), 
                    high=np.array([[10000.0, 300, 2]] * totalTrafficLights),
                    shape=(totalTrafficLights, 3,), 
                    dtype=np.float32
                )
            )
        
    """
        GymEnv = Union[gym.Env, vec_env.VecEnv]
        GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
        GymResetReturn = Tuple[GymObs, Dict]
        GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
        in this case the return value would be a tuple defined as follows
        tuple(spaces.Tuple, float, bool, bool, int):
    """
    # def step(self, action): action??
    # update actions first, then get_obs
    def step(self) -> GymStepReturn: 
        self.tick()
        observation = self._get_observation()
        self.rewardMap.tick()
        reward = self.rewardMap.getReward() # update reward and reward map
        # unnecessary to check max_step as ev will always arrive destination
        terminated = bool(self.getLocation() >= 10000.0)
        truncated = False # unnecessary to truncate anything
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) 
            -> tuple[spaces.Tuple[spaces.Discrete, space.Box], info]:
        # should re-initialize all traffic light status 
        # reset all rewards ?
        # and place the ego_vehicle to correct location ?

        # return value `info` is not currently used, set to None for now
        World.reset(self)
        observation = _get_observation()
        info = {}
        return observation, info

    def render(self):
        # not implemented in this simulation
        # will use CARLA for demo
        pass

    def close(self):
        pass

    def _get_observation() -> spaces.Tuple:
        # need a pointer in World class that points to ego_vehicle
        ego_vehicle = self.get_ego_vehicle()
        lights_status = []
        for a in self.actors:
            if(a._find_Actor_Type() == "TrafficLight"):
                lights_status.append([a.getLocation(),
                                      a.getCountdown(),
                                      a.getPhase()])
        observation = spaces.Tuple(
            np.array([ego_vehicle.getLocation(), ]),
            np.array([ego_vehicle.getSpeed()], ),
            self.numTrafficLightAhead(ego_vehicle),
            lights_status
        )
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