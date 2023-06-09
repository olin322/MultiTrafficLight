# from World import World
# from Actor import Actor
import matplotlib.pyplot as plt
import random

# world = World(0.02)
# for i in range(100):
# 	world.tick()

ydata = []
xdata = []
fig, ax = plt.subplots()


for i in range(100):
	xdata.insert(-1, i)
	ydata.insert(-1, random.randint(0,9))
	ax.plot(xdata, ydata)
	plt.show()
