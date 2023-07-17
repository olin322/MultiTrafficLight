import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StraightRoadEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": ["human"], 
        "render_fps": 30
    }

    def __init__(self, totalTrafficLights: int, ):
        super().__init__()
        
        
        """
        | Num |                 Action                   | Control Min | Control Max | Unit |
        |  0  | accelerate at maximum acceleration       |     N/A     |     N/A     |      |
        |  1  | deaccelerate at maximum deacceleration   |     N/A     |     N/A     |      |
        """
        self.action_space = spaces.Discrete(2)


        """
        | Num |                 Observation                          | Value Min |   Vale Max   | Unit  |
        |  0  | location of ego_vehicle at previous step             |     0     |     10000    |   m   |
        |  1  | speed/velocity of ego_vehicle at previous step       |     0     |     16.67    |  m/s  |
        |  2  | number of traffic ligth ahead                        |           |              |       |
        |     |

        """
        self.observation_space = spaces.Tuple(
                                            space.Discrete(2),
                                            Box(
                                                low=0, high=10000,
                                                shape=(), 
                                                dtype=np.float32
                                            )
                                        )
        

    def step(self, action):
        pass
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        
        return observation, info

    def render(self):
        pass

    def close(self):
        pass