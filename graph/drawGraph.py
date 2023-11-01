import matplotlib.pyplot as plt
import numpy as np




def drawSeventeenTrafficLights():
	from location24b import location24b
	from locations_full import sl_full
	# l = []
	# timel = []
	# for i in range(3306):
	# 	timel.append(float(i)/10) # delta_t = 0.1
	# 	l.append(locations[i]/100)
	l_24b = []
	timel_24b = []
	for i in range(10001):
		timel_24b.append(float(i)/10)
		l_24b.append(location24b[i]/100)
	plt.plot(timel_24b, l_24b, color='orange')
	# plt.plot(timel, l, color='orange')
	sl = []
	timesl = []
	for i in range(50515):
		timesl.append(float(i)/50) # delta_t = 50
		sl.append(sl_full[i]/100)
	plt.plot(timesl, sl, color='blue')
	# plt.scatter(time, locations)
	light_number = [
					'light 1  100', 
					'light 2  200',
					'light 3  500',
					'light 4 2000',
					'light 5 2500',
					'light 6 3200',
					'light 7 3400',
					'light 8 3600',
					'light 9 3800',
					'light10 4000',
					'light11 5000',
					'light12 5100',
					'light13 6000',
					'light14 6500',
					'light15 7000',
					'light16 8000',
					'light17 9900'
					]
	pos = [ 1,  2,  5, 20, 25, 32, 34, 36, 38, 40, 
		   50, 51, 60, 65, 70, 80, 99]
	pos = np.multiply(pos, 100)
	cycle   = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57,
						57, 67, 50, 61, 61, 61, 61])
	spat = []
	for i in range(17):
		spat.append([])
	print(spat)
	plt.yticks(pos, light_number)

	plt.show()

drawSeventeenTrafficLights()

def myexample():
	from locations import locations
	from simpleLocations import simpleLocations
	
	l = []
	timel = []
	for i in range(3306):
		timel.append(float(i)/10) # delta_t = 10
		l.append(locations[i]/100)
	l_14b = []
	timel_14b = []
	for i in range(3485):
		timel_14b.append(float(i)/10)
		l_14b.append(location14b[i]/100)
	plt.plot(timel_14b, l_14b, color='orange')
	# plt.plot(timel, l, color='orange')
	sl = []
	timesl = []
	for i in range(17701):
		timesl.append(float(i)/50) # delta_t = 50
		sl.append(simpleLocations[i]/100)
	plt.plot(timesl, sl, color='blue')
	# plt.scatter(time, locations)
	light_number = [
					'light 1  100', 
					'light 2  200',
					'light 3  500',
					'light 4 2000',
					'light 5 2500',
					'light 6 3200',
					'light 7 3400',
					'light 8 3600',
					'light 9 3800',
					'light10 4000',]
	pos = [1, 2, 5, 20, 25, 32, 34, 36, 38, 40]
	# pos = np.multiply(pos, 100)
	yellow  = np.array([ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3]) 
	cycle_1 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57]) # g
	#yellow = np.array([ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3]) # y	
	cycle_2 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57]) # r
	cycle_3 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57]) # g
	#yellow = np.array([ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3]) # y
	cycle_4 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57]) # r
	cycle_5 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57]) # g 341
	#yellow = np.array([ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3])
	# cycle_6 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57]) # 411
	# cycle_7 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 57])
	#yellow = np.array([ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3]) # y 
	cyclee1 = np.array([10, 47, 61, 53, 53, 61, 56, 56, 56, 57]) # r 400
	cyclee2 = np.array([10, 47, 25, 53, 53, 25,  0,  0,  0, 49]) # g
	yellow2 = np.array([ 3,  3,  0,  3,  3,  0,  0,  0,  0,  0]) # y
	cyclee3 = np.array([10, 47,  0, 17, 17,  0,  0,  0,  0,  0])
	cyclee4 = np.array([10, 12,  0,  0,  0,  0,  0,  0,  0,  0]) # 
	cyclee5 = np.array([10,  0,  0,  0,  0,  0,  0,  0,  0,  0]) # 
	yellow3 = np.array([ 3,  0,  0,  0,  0,  0,  0,  0,  0,  0])

	plt.barh(pos, cycle_1, color='g')
	plt.barh(pos, yellow,  left=cycle_1, color='y')
	plt.barh(pos, cycle_2, left=cycle_1+yellow, color='r')
	plt.barh(pos, cycle_3, left=cycle_1+yellow +cycle_2, color='g')
	plt.barh(pos, yellow,  left=cycle_1+yellow +cycle_2+cycle_3, color='y')
	plt.barh(pos, cycle_4, left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow , color='r')
	plt.barh(pos, cycle_5, left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow +cycle_4, color='g')
	plt.barh(pos, yellow,  left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow +cycle_4+cycle_5, color='y')
	# plt.barh(pos, cycle_6, left=cycle_1+yellow +cycle_2+cycle_3
	# 						   +yellow +cycle_4+cycle_5+yellow, color='r')
	# plt.barh(pos, cycle_7, left=cycle_1+yellow +cycle_2+cycle_3
	# 						   +yellow +cycle_4+cycle_5+yellow
	# 						   +cycle_6, color='g')
	# plt.barh(pos, yellow,  left=cycle_1+yellow +cycle_2+cycle_3
	# 						   +yellow +cycle_4+cycle_5+yellow
	# 						   +cycle_6+cycle_7, color='y')
	plt.barh(pos, cyclee1,  left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow +cycle_4+cycle_5+yellow, color='r')
	plt.barh(pos, cyclee2,  left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow +cycle_4+cycle_5+yellow
							   +cyclee1, color='g')
	plt.barh(pos, yellow2,   left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow +cycle_4+cycle_5+yellow
							   +cyclee1+cyclee2, color='y')
	plt.barh(pos, cyclee3,   left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow +cycle_4+cycle_5+yellow
							   +cyclee1+cyclee2+yellow2, color='r')
	plt.barh(pos, cyclee4,   left=cycle_1+yellow +cycle_2+cycle_3
							   +yellow +cycle_4+cycle_5+yellow
							   +cyclee1+cyclee2+yellow2+cyclee3, color='g')
	for i in range(0,12,1):
		plt.barh(pos, yellow3,  left=cycle_1+yellow+cycle_2+cycle_3
								+yellow +cycle_4+cycle_5+yellow
								+cyclee1+cyclee2+yellow2+cyclee3
								+cyclee4+np.multiply(cyclee5, i*2)+np.multiply(yellow3, i), color='y')
		t = np.multiply(cyclee5, i*2)
		plt.barh(pos, cyclee5,  left=cycle_1+yellow +cycle_2+cycle_3
								+yellow +cycle_4+cycle_5+yellow
								+cyclee1+cyclee2+yellow2+cyclee3
								+cyclee4+np.multiply(yellow3, i+1)+t, color='r')
		t = np.multiply(cyclee5, i*2+1)
		plt.barh(pos, cyclee5,  left=cycle_1+yellow +cycle_2+cycle_3
								+yellow +cycle_4+cycle_5+yellow
								+cyclee1+cyclee2+yellow2+cyclee3
								+cyclee4+np.multiply(yellow3, i+1)+t, color='g')
	plt.barh(pos, yellow3,  left=cycle_1+yellow+cycle_2+cycle_3
							+yellow +cycle_4+cycle_5+yellow
							+cyclee1+cyclee2+yellow2+cyclee3
							+cyclee4+np.multiply(cyclee5, 12*2)+np.multiply(yellow3, 12), color='y')
	t = np.multiply(cyclee5, 12*2)
	plt.barh(pos, cyclee5,  left=cycle_1+yellow +cycle_2+cycle_3
							+yellow +cycle_4+cycle_5+yellow
							+cyclee1+cyclee2+yellow2+cyclee3
							+cyclee4+np.multiply(yellow3, 12+1)+t, color='r')
	t = np.multiply(cyclee5, 12*2+1)
	plt.barh(pos, [9, 0, 0, 0, 0, 0, 0, 0, 0, 0],  left=cycle_1+yellow +cycle_2+cycle_3
							+yellow +cycle_4+cycle_5+yellow
							+cyclee1+cyclee2+yellow2+cyclee3
							+cyclee4+np.multiply(yellow3, 12+1)+t, color='g')
	plt.yticks(pos, light_number)

	plt.show()

# myexample()

def example():
	
	# create dataset
	height = [3, 12, 5, 18, 45]
	bars = ('A', 'B', 'C', 'D', 'E')
	 
	# Choose the position of each barplots on the x-axis (space=1,4,3,1)
	x_pos = [0,1,5,8,9]
	 
	# Create bars
	plt.barh(x_pos, height)
	 
	# Create names on the x-axis
	plt.yticks(x_pos, bars)
	 
	# Show graphic
	plt.show()

# example()