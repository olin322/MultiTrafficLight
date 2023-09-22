"""
test test
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from overrides import override

class Animal:
    def __init__(self):
        print("is Animal")

    def speak(self):
        print("Animal speaks")

    def funcA(self):
        print("Animal funcA")

class Mamal:
    def __init__(self):
        print("is Mamal")

class Cat(Animal, Mamal):
    def __init__(self):
        super().__init__()
        print("\n and Cat")
    
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

# animals = [Cat(), Dog(), Robot()]

a = Cat()
# a.funcC()
# for animal in animals:
    # animal.speak()

