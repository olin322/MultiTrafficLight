"""
test test
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from typing import Override

class Animal:
    def speak(self):
        print("Animal speaks")

class Cat(Animal):
    # @override
    def speak(self):
        print("Meow")

class Dog(Animal):
    # @override
    def speak(self):
        print("Woof")

class Robot:
    def speak(self):
        print("Beep boop")

animals = [Cat(), Dog(), Robot()]

for animal in animals:
    animal.speak()

