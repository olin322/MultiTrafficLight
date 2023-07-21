"""
test test
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from overrides import override

class Animal:
    def speak(self):
        print("Animal speaks")

    def funcA(self):
        print("Animal funcA")

class Cat(Animal):
    @override
    def speak(self):
        print("Meow")

    def funcB(self):
        self.funcA()

    @override
    def funcA(self):
        print("funcA in Cat")

    def funcC(self):
        super().funcA()

class Dog(Animal):
    @override
    def speak(self):
        print("Woof")

class Robot:
    def speak(self):
        print("Beep boop")

animals = [Cat(), Dog(), Robot()]

a = Cat()
a.funcC()
# for animal in animals:
    # animal.speak()

