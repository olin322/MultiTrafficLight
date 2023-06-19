import World
from World import World
from Actor import Actor
from Vehicle import Vehicle
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import math

### Currently the simulation runs in 1-D space/x-axis
### CONSTANTS
# speed_limit = 60km/h
DESTINATION = 100 # m
world = World(0.02)
ego_vehicle = Vehicle(0.0, 1500.0, 7.9, 7.9, world.get_delta_t())
world.spawn_vehicle(ego_vehicle)
while(ego_vehicle.getLocation() < DESTINATION):
	world.tick()
print("simulation time = ", world.get_simulation_time())

### Animation
# ydata = []
# xdata = []
# fig, ax = plt.subplots()

# def animate(i):
# 	xdata.append(random.randint(0,9))
# 	ydata.append(random.randint(0,9))
# 	x_min = min(xdata)
# 	x_max = max(xdata)
# 	y_min = min(ydata)
# 	y_max = max(ydata)
# 	# fig, ax = plt.subplots()
# 	ax.set_xlim(x_min - 1, x_max + 1)
# 	ax.set_ylim(y_min - 1, y_max + 1)
# 	line, = ax.plot(xdata, ydata, 'ro')
# 	line.set_ydata(ydata)
# 	return line,

# ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count = 50)
# plt.show()


# fig, ax = plt.subplots()
# ax.set_xlim(0, 10) 
# ax.set_ylim(0, 2)

# x_data, y_data = [], []

# def animate(i): 
#     x_data.append(i) 
#     y_data.append(math.sin(math.pi*i)) 

#     line, = ax.plot(x_data, y_data, 'r') 
#     line.set_ydata(y_data)
#     return line,

# ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count = 50)
# plt.show()

















##############################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig, ax = plt.subplots()

# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))

# def animate(i):
#     line.set_ydata(np.sin(x + i / 50))  # update the data.
#     return line,

# ani = animation.FuncAnimation(
#     fig, animate, interval=20, blit=True, save_count=50) 

# # To save the animation, use e.g.
# #
# # ani.save("movie.mp4")
# #
# # or
# #
# # writer = animation.FFMpegWriter(
# #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# # ani.save("movie.mp4", writer=writer)

# plt.show()









# import numpy as np
# import matplotlib.pyplot as plt 
# import matplotlib.animation as animation

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# fig, ax = plt.subplots()
# x = np.arange(0, 2*np.pi, 0.01)        # x-array

# def animate(i): 
#     line.set_ydata(np.sin(x + i/10.0))  # update the data
#     return line, 

# # Init only required for blitting to give a clean slate.
# def init():  
#     line.set_ydata(np.sin(x)) 
#     return line,

# line, = ax.plot(x, np.sin(x))
# ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,  
#                               interval=25, blit=True)
# plt.show()
