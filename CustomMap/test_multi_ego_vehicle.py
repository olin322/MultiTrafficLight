
import carla
from carla import Actor
from carla import Vector3D
from carla import Transform, Location, Rotation

# from stable_baselines3.common.env_checker import check_env

# from World import World
# from Actor import Actor
# from Vehicle import Vehicle
# from rewards import RewardMap
# from TrafficLight import TrafficLight
# from envs.straightRoad import StraightRoadEnv

# import matplotlib.pyplot as plt
# import random
# import matplotlib.animation as animation
# import math
# from datetime import date
# from datetime import datetime
# from datetime import timedelta

# import gymnasium as gym

# from stable_baselines3 import SAC, TD3, A2C, DDPG
# from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.noise import NormalActionNoise



client = carla.Client('localhost', 2000)
client.load_world('Town06')
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.02
world.apply_settings(settings)# level = world.get_map()
blueprint_library = world.get_blueprint_library()

ego_vehicle_spawn_point_1 = carla.Transform(
  carla.Location(x=-272, y=-18, z=0.281494), 
  carla.Rotation(pitch=0.000000, yaw=0.0, roll=0.000000)
)
ego_vehicle_spawn_point_2 = carla.Transform(
  carla.Location(x=-272, y=-10, z=0.281494), 
  carla.Rotation(pitch=0.000000, yaw=0.0, roll=0.000000)
)

ego_vehicle_1 = world.spawn_actor(
    blueprint_library.find('vehicle.lincoln.mkz_2020'), 
    ego_vehicle_spawn_point_1
)
ego_vehicle_2 = world.spawn_actor(
    blueprint_library.find('vehicle.tesla.model3'), 
    ego_vehicle_spawn_point_2
)

spectator = world.get_spectator()
spectator_transform = ego_vehicle_1.get_transform()
spectator.set_transform(
    carla.Transform(spectator_transform.location + \
        carla.Location(x=0, y=0,z=30), carla.Rotation(pitch=-90)
    )
)

# frame = snapshot.frame - frame_zero
elapsed_seconds = 0
step = 1
while (step <= 20000):
    step += 1
    snapshot = world.get_snapshot()
    elapsed_seconds = snapshot.elapsed_seconds
    ego_vehicle_throttle_1 = 0.2
    ego_vehicle_steer_1 = 0.0
    ego_vehicle_1.apply_control(
        carla.VehicleControl(
            throttle=ego_vehicle_throttle_1, 
            steer=ego_vehicle_steer_1
        )
    )
    ego_vehicle_throttle_2 = 0.11
    ego_vehicle_steer_2 = 0.0
    ego_vehicle_2.apply_control(
        carla.VehicleControl(
            throttle=ego_vehicle_throttle_2, 
            steer=ego_vehicle_steer_2
        )
    )
    spectator = world.get_spectator()
    spectator_transform = ego_vehicle_1.get_transform()
    spectator.set_transform(
        carla.Transform(spectator_transform.location + \
            carla.Location(x=0, y=0,z=30), carla.Rotation(pitch=-90)
        )
    )
    world.tick()
    

#############################################################################################
#
# #!/usr/bin/env python
#
# import sys
#
# sys.path.append(
#     'PythonAPI/dist/carla-0.9.14-py3.10-linux-x86_64.egg' % (sys.version_info.major,
#                                                              sys.version_info.minor))
#
# import carla
#
############################################################################################
