<!--
This 
is 
multi-line
comment
-->




# Study Notes - Stable Baselines3

## Using Custom Environments

##### Basic Framework
```
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, arg1, arg2, ...):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...
```

##### Define and train a RL agent
```
# Instantiate the env
env = CustomEnv(arg1, ...)
# Define and Train the agent
model = A2C("CnnPolicy", env).learn(total_timesteps=1000)
```


##### Check custom environment follows the Gym interface
```
from stable_baselines3.common.env_checker import check_env

env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)
```

<!-- <div style="page-break-after: always;"></div> -->

### Register the environment with gym

this will allow you to create the RL agent in one line (and use `gym.make()` to instantiate the env):
```
from gymnasium.envs.registration import register
# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="CartPole-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="gym.envs.classic_control:CartPoleEnv", # note this is relative path
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)
```

<!-- <div style="page-break-after: always;"></div> -->

##### A example of custom environment named `IdentityEnv` and how to use it
```
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.type_aliases import GymStepReturn

T = TypeVar("T", int, np.ndarray)


class IdentityEnv(gym.Env, Generic[T]):
    def __init__(self, dim: Optional[int] = None, space: Optional[spaces.Space] = None, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of ``dim`` and ``space``. If both are
            None, then initialization proceeds with ``dim=1`` and ``space=None``.
        :param space: the action and observation space. Provide at most one of
            ``dim`` and ``space``.
        :param ep_length: the length of each episode in timesteps
        """
        if space is None:
            if dim is None:
                dim = 1
            space = spaces.Discrete(dim)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        self.action_space = self.observation_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[T, Dict]:
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state, {}

    def step(self, action: T) -> Tuple[T, float, bool, bool, Dict[str, Any]]:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.ep_length
        return self.state, reward, terminated, truncated, {}

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def _get_reward(self, action: T) -> float:
        return 1.0 if np.all(self.state == action) else 0.0

    def render(self, mode: str = "human") -> None:
        pass


class IdentityEnvBox(IdentityEnv[np.ndarray]):
    def __init__(self, low: float = -1.0, high: float = 1.0, eps: float = 0.05, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param low: the lower bound of the box dim
        :param high: the upper bound of the box dim
        :param eps: the epsilon bound for correct value
        :param ep_length: the length of each episode in timesteps
        """
        space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
        super().__init__(ep_length=ep_length, space=space)
        self.eps = eps

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.ep_length
        return self.state, reward, terminated, truncated, {}

    def _get_reward(self, action: np.ndarray) -> float:
        return 1.0 if (self.state - self.eps) <= action <= (self.state + self.eps) else 0.0


class IdentityEnvMultiDiscrete(IdentityEnv[np.ndarray]):
    def __init__(self, dim: int = 1, ep_length: int = 100) -> None:
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        space = spaces.MultiDiscrete([dim, dim])
        super().__init__(ep_length=ep_length, space=space)


class IdentityEnvMultiBinary(IdentityEnv[np.ndarray]):
    def __init__(self, dim: int = 1, ep_length: int = 100) -> None:
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        space = spaces.MultiBinary(dim)
        super().__init__(ep_length=ep_length, space=space)


class FakeImageEnv(gym.Env):
    """
    Fake image environment for testing purposes, it mimics Atari games.

    :param action_dim: Number of discrete actions
    :param screen_height: Height of the image
    :param screen_width: Width of the image
    :param n_channels: Number of color channels
    :param discrete: Create discrete action space instead of continuous
    :param channel_first: Put channels on first axis instead of last
    """

    def __init__(
        self,
        action_dim: int = 6,
        screen_height: int = 84,
        screen_width: int = 84,
        n_channels: int = 1,
        discrete: bool = True,
        channel_first: bool = False,
    ) -> None:
        self.observation_shape = (screen_height, screen_width, n_channels)
        if channel_first:
            self.observation_shape = (n_channels, screen_height, screen_width)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        if discrete:
            self.action_space = spaces.Discrete(action_dim)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.ep_length = 10
        self.current_step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        return self.observation_space.sample(), {}

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        reward = 0.0
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.ep_length
        return self.observation_space.sample(), reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        pass
```

##### and how to use the above custom env
```
import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.envs import IdentityEnv, IdentityEnvBox, IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

DIM = 4


@pytest.mark.parametrize("model_class", [A2C, PPO, DQN])
@pytest.mark.parametrize("env", [IdentityEnv(DIM), IdentityEnvMultiDiscrete(DIM), IdentityEnvMultiBinary(DIM)])
def test_discrete(model_class, env):
    env_ = DummyVecEnv([lambda: env])
    kwargs = {}
    n_steps = 2500
    if model_class == DQN:
        kwargs = dict(learning_starts=0)
        # DQN only support discrete actions
        if isinstance(env, (IdentityEnvMultiDiscrete, IdentityEnvMultiBinary)):
            return

    model = model_class("MlpPolicy", env_, gamma=0.4, seed=3, **kwargs).learn(n_steps)

    evaluate_policy(model, env_, n_eval_episodes=20, reward_threshold=90, warn=False)
    obs, _ = env.reset()

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, DDPG, TD3])
def test_continuous(model_class):
    env = IdentityEnvBox(eps=0.5)

    n_steps = {A2C: 2000, PPO: 2000, SAC: 400, TD3: 400, DDPG: 400}[model_class]

    kwargs = dict(policy_kwargs=dict(net_arch=[64, 64]), seed=0, gamma=0.95)

    if model_class in [TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise
    elif model_class in [A2C]:
        kwargs["policy_kwargs"]["log_std_init"] = -0.5
    elif model_class == PPO:
        kwargs = dict(n_steps=512, n_epochs=5)

    model = model_class("MlpPolicy", env, learning_rate=1e-3, **kwargs).learn(n_steps)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)
```












<div style="page-break-after: always;"></div>

#### Multi-Processing

```
    # note using multi-processing does not make training faster
    # as stated in the answer to [this github issue](https://github.com/hill-a/stable-baselines/issues/1113)


    # Is it true that having multiple envs even though running sequentially, 
    # will make the training stable?

    # At the end, both are synchronous, 
    # so it does not change anything for the agent if you use 
    # a DummyVecEnv with 4 envs or a SubprocVecEnv with 4 envs. 
    # What may change is the fps (cf. notebook for a comparison).
```

the following code should run 
```
import gymnasium as gym
from stable_baselines3 import SAC

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# below is an example
def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "CartPole-v1"
    num_process = 8 # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_process)])
    
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    model = SAC("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25_000)
    
    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()

```


<!--
<script 
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
    type="text/javascript">
</script> 
-->

#### Normalizing Input Features on `PyBullet` environments
```
pip install pybullet
```

```
import os
import gymnasium as gym
import pybullet_envs
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
# Note: pybullet is not compatible yet with Gymnasium
# you might need to use `import rl_zoo3.gym_patches`
# and use gym (not Gymnasium) to instantiate the env
# Alternatively, you can use the MuJoCo equivalent "HalfCheetah-v4"
vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
# Automatically normalize the input features and reward
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
clip_obs=10.)
model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps=2000)
# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "/tmp/"
model.save(log_dir + "ppo_halfcheetah")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)

# To demonstrate loading
del model, vec_env
# Load the saved statistics
vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
vec_env = VecNormalize.load(stats_path, vec_env)
# do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False
# Load the agent
model = PPO.load(log_dir + "ppo_halfcheetah", env=vec_env)
```



#### Learning Rate Schedule (Adjust Learning Rate $$\alpha$$)

```
from typing import Callable
from stable_baselines3 import PPO

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func
# Initial learning rate of 0.001
model = PPO("MlpPolicy", "CartPole-v1", learning_rate=linear_schedule(0.001), verbose=1)
model.learn(total_timesteps=20_000)
# By default, `reset_num_timesteps` is True, in which case the learning rate schedule resets.
# progress_remaining = 1.0 - (num_timesteps / total_timesteps)
model.learn(total_timesteps=10_000, reset_num_timesteps=True)

```






### Saving and Loading Model & Policy
>> Note that when loading a saved model, 
>> it is required to do `env = ModelsEnv()` and `model.set_env(env)`
```
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy
# Create the model and the training environment
model = SAC("MlpPolicy", "Pendulum-v1", verbose=1,
learning_rate=1e-3)
# train the model
model.learn(total_timesteps=6000)
# save the model
model.save("sac_pendulum")
# the saved model does not contain the replay buffer
loaded_model = SAC.load("sac_pendulum")
print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its
buffer")
# now save the replay buffer too
model.save_replay_buffer("sac_replay_buffer")
# load it into the loaded_model
loaded_model.load_replay_buffer("sac_replay_buffer")
# now the loaded replay is not empty anymore
print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")
# Save the policy independently from the model
# Note: if you don't save the complete model with `model.save()`
# you cannot continue training afterward
policy = model.policy
policy.save("sac_policy_pendulum")
# Retrieve the environment
env = model.get_env()
# Evaluate the policy
mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
# Load the policy independently from the model
saved_policy = MlpPolicy.load("sac_policy_pendulum")
# Evaluate the loaded policy
mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```