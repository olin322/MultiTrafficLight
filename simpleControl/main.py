from World import World
from Actor import Actor
from Vehicle import Vehicle
from TrafficLight import TrafficLight
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import math
from datetime import date
from datetime import datetime
from datetime import timedelta
from overrides import override

### Currently the simulation runs in 1-D space/x-axis
### CONSTANTS
# speed_limit = 60km/h


# hyper parameter
frame = 0
DESTINATION = 10000 # m

# num = 1

world = World(0.02)
ego_vehicle     = Vehicle(0.0, 1500.0, 2, 2, world.get_delta_t())
trafficLight_1  = TrafficLight(100,  "green", 10, world.get_delta_t())
trafficLight_2  = TrafficLight(200,  "green", 47, world.get_delta_t())
trafficLight_3  = TrafficLight(500,  "green", 61, world.get_delta_t())
trafficLight_4  = TrafficLight(2000, "green", 53, world.get_delta_t())
trafficLight_5  = TrafficLight(2500, "green", 53, world.get_delta_t())
trafficLight_6  = TrafficLight(3200, "green", 61, world.get_delta_t())
trafficLight_7  = TrafficLight(3400, "green", 67, world.get_delta_t())
trafficLight_8  = TrafficLight(3600, "green", 67, world.get_delta_t())
trafficLight_9  = TrafficLight(3800, "green", 67, world.get_delta_t())
trafficLight_10 = TrafficLight(4000, "green", 57, world.get_delta_t())
trafficLight_11 = TrafficLight(5000, "green", 57, world.get_delta_t())
trafficLight_12 = TrafficLight(5100, "green", 67, world.get_delta_t())
trafficLight_13 = TrafficLight(6000, "green", 61, world.get_delta_t())
trafficLight_14 = TrafficLight(7000, "green", 61, world.get_delta_t())
trafficLight_15 = TrafficLight(8000, "green", 61, world.get_delta_t())
trafficLight_16 = TrafficLight(9900, "green", 61, world.get_delta_t())


def main(): 
  global frame, num
    
  world.spawn_vehicle(ego_vehicle)
  world.add_traffic_light(trafficLight_1)
  world.add_traffic_light(trafficLight_2)
  world.add_traffic_light(trafficLight_3)
  world.add_traffic_light(trafficLight_4)
  world.add_traffic_light(trafficLight_5)
  world.add_traffic_light(trafficLight_6)
  world.add_traffic_light(trafficLight_7)
  world.add_traffic_light(trafficLight_8)
  world.add_traffic_light(trafficLight_9)
  world.add_traffic_light(trafficLight_10)
  world.add_traffic_light(trafficLight_11)
  world.add_traffic_light(trafficLight_12)
  world.add_traffic_light(trafficLight_13)
  world.add_traffic_light(trafficLight_14)
  world.add_traffic_light(trafficLight_15)
  world.add_traffic_light(trafficLight_16)


  # debug
  log_data = ""
  log_name = get_debug_log_name()
  while ((ego_vehicle.getLocation() >= 0) & 
      (ego_vehicle.getLocation() < DESTINATION)):
    # print(frame)
    frame += 1
    log_debug_data(log_name)
    world.tick()

    # print("simulation time = ", world.get_simulation_time())

    # print("speed = ", ego_vehicle.getSpeed())
    # print("light count down: ", trafficLight_1.getCountdown(), " ", trafficLight_1.getPhase())

def log_debug_data(log_name: str) -> None:
  log_data = get_debug_log_data()
  f = open(log_name, 'a')
  f.write(log_data)
  f.close()
  return None

def get_debug_log_name() -> str:
  today = date.today()
  current_time = datetime.now().strftime("%H:%M:%S")
  return(str(today)+"-"+str(current_time)+".txt")

def get_debug_log_data() -> str:
  log_data = "frame = " + str(frame) + "\t"\
        + " sim time = " + _roundup(world.get_simulation_time(), 6) + "\t"\
        + " ev speed = " + _roundup(ego_vehicle.getSpeed(), 6) + "\t"\
        + " ev location = " + _roundup(ego_vehicle.getLocation(), 6) + "\t"\
        
  if(world._find_next_light()):
      log_data += " countdown = " + str(world._find_next_light().getCountdown())\
          + " " + world._find_next_light().getPhase() \
          + " \tnext light location: " + _roundup(world._find_next_light().getLocation(), 6) \
          + " \treal world time stamp: " + str(datetime.utcnow() + timedelta(hours=8)) + "\n"
  else:
    log_data += " vehicle passed all traffic lights"\
        + "\treal world time stamp: " + str(datetime.utcnow() + timedelta(hours=8))\
        + "\n"  
  return log_data

def _roundup(d: float, l: int) -> str:
  s = str(round(d, l))
  i = 0
  if (s.find(".") != -1):
    i = s.find(".")
  else:
    return str(d)
  while((len(s) - i) <= l):
    s += '0'
  return s

main()










############################################################################



### Animation
# ydata = []
# xdata = []
# fig, ax = plt.subplots()

# def animate(i):
#   xdata.append(random.randint(0,9))
#   ydata.append(random.randint(0,9))
#   x_min = min(xdata)
#   x_max = max(xdata)
#   y_min = min(ydata)
#   y_max = max(ydata)
#   # fig, ax = plt.subplots()
#   ax.set_xlim(x_min - 1, x_max + 1)
#   ax.set_ylim(y_min - 1, y_max + 1)
#   line, = ax.plot(xdata, ydata, 'ro')
#   line.set_ydata(ydata)
#   return line,

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