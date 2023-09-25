#!/usr/bin/env python

import sys
from mapinfo import MAP

sys.path.append(
    '~Owen/CarlaFromSource/carla/PythonAPI/carla/dist/carla-X.X.X-py%d.%d-linux-x86_64.egg' % (sys.version_info.major,
                                                             sys.version_info.minor))

import math
import carla
from carla import Actor
from carla import Vector3D
from datetime import date
from datetime import datetime
from mapinfo import MAP

client = carla.Client('localhost', 2000)
# load map 
client.generate_opendrive_world(MAP)
# client.load_world('Town10HD_Opt')
world = client.get_world()
settings = world.get_settings()

step = 0

settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.02
world.apply_settings(settings)# level = world.get_map()
blueprint_library = world.get_blueprint_library()

ego_vehicle_spawn_point_1 = carla.Transform(
  carla.Location(x=-4990, y=3, z=0.2), 
  carla.Rotation(pitch=0.000000, yaw=0.0, roll=0.000000)
)
ego_vehicle_spawn_point_2 = carla.Transform(
  carla.Location(x=-4990, y=-3, z=0.2), 
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

# spectator = world.get_spectator()
# spectator_transform = ego_vehicle_1.get_transform()
# spectator.set_transform(
#     carla.Transform(spectator_transform.location + \
#         carla.Location(x=0, y=0,z=30), carla.Rotation(pitch=-90)
#     )
# )





# ego_vehicle_spawn_point = carla.Transform(carla.Location(x=-272, y=-18, z=0.281494), \
#                                           carla.Rotation(pitch=0.000000, yaw=0.0, roll=0.000000))
# ego_vehicle = world.spawn_actor(blueprint_library.find('vehicle.tesla.model3'), ego_vehicle_spawn_point)

# settings.synchronous_mode = True # Enables synchronous mode
# settings.fixed_delta_seconds = 0.02
# world.apply_settings(settings)

spectator = world.get_spectator()
spectator_transform = ego_vehicle_1.get_transform()
spectator.set_transform(
    carla.Transform(
        carla.Location(x=-5000, y=0,z=30), carla.Rotation(pitch=-90)
    )
)

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