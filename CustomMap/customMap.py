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

# ego_vehicle_spawn_point = carla.Transform(carla.Location(x=-272, y=-18, z=0.281494), \
#                                           carla.Rotation(pitch=0.000000, yaw=0.0, roll=0.000000))
# ego_vehicle = world.spawn_actor(blueprint_library.find('vehicle.tesla.model3'), ego_vehicle_spawn_point)

# settings.synchronous_mode = True # Enables synchronous mode
# settings.fixed_delta_seconds = 0.02
# world.apply_settings(settings)

spectator = world.get_spectator()
# spectator_transform = ego_vehicle_1.get_transform()
spectator.set_transform(
    carla.Transform(
        carla.Location(x=-5000, y=0,z=30), carla.Rotation(pitch=-90)
    )
)