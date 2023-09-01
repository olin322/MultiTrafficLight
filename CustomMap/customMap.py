#!/usr/bin/env python

import sys

# sys.path.append('~Owen/CarlaFromSource/carla/PythonAPI/carla/dist/carla-0.9.14-py3.10-linux-x86_64.egg' % (sys.version_info.major, sys.version_info.minor))
sys.path.append(
    '~Owen/CarlaFromSource/carla/PythonAPI/carla/dist/carla-X.X.X-py%d.%d-linux-x86_64.egg' % (sys.version_info.major,
                                                             sys.version_info.minor))

import math
import carla
from carla import Actor
from carla import Vector3D
from datetime import date
from datetime import datetime


client = carla.Client('localhost', 2000)
# load map 
client.load_world('10kmStaightRoad')
world = client.get_world()
settings = world.get_settings()

settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.02
world.apply_settings(settings)