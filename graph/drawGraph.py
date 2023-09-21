import matplotlib.pyplot as plt
import numpy as np

def myexample():
	light_number = [
					'light 1', 
					'light 2',
					'light 3',
					'light 4',
					'light 5',
					'light 6',
					'light 7',
					'light 8',
					'light 9',
					'light10',]
	pos = [1, 2, 5, 20, 25, 32, 34, 36, 38, 40]
	cycle_1 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 51])	
	cycle_2 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 51])
	cycle_3 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 51])
	cycle_4 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 51])
	cycle_5 = np.array([10, 47, 61, 53, 53, 61, 67, 67, 67, 51])
	cycle_6 = [10, 47, 61, 53, 53, 61, 67, 67, 67, 51]
	cycle_7 = [10, 47, 61, 53, 53, 61, 67, 67, 67, 51] # 67 * 7 = 469
	cyclee1 = [10, 47, 42, 53, 53, 42,  0,  0,  0, 51]
	cyclee2 = [10, 47,  0, 45, 45,  0,  0,  0,  0, 51]
	cyclee3 = [10, 46,  0,  0,  0,  0,  0,  0,  0, 10]
	cyclee4 = [10,  0,  0,  0,  0,  0,  0,  0,  0,  0] # 11
	cyclee5 = [10,  0,  0,  0,  0,  0,  0,  0,  0,  0] # 11

	plt.barh(pos, cycle_1, color='r')
	plt.barh(pos, cycle_2, left=cycle_1, color='g')
	plt.barh(pos, cycle_3, left=cycle_1+cycle_2, color='r')
	plt.barh(pos, cycle_4, left=cycle_1+cycle_2+cycle_3, color='g')
	plt.barh(pos, cycle_5, left=cycle_1+cycle_2+cycle_3+cycle_4, color='r')
	plt.barh(pos, cycle_6, left=cycle_1+cycle_2+cycle_3
							   +cycle_4+cycle_5, color='g')
	plt.barh(pos, cycle_7, left=cycle_1+cycle_2+cycle_3
							   +cycle_4+cycle_5+cycle_6, color='r')
	plt.barh(pos, cyclee1, left=cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7, color='g')
	plt.barh(pos, cyclee2, left=cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7+cyclee1, color='r')
	plt.barh(pos, cyclee3, left=cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7+cyclee1+cyclee2, color='g')
	plt.barh(pos, cyclee4, left=cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7+cyclee1+cyclee2+cyclee3, color='r')
	for i in range(0,17,1):
		t = np.multiply(cyclee4, i*2+1)
		plt.barh(pos, cyclee5, left=(cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7+cyclee1+cyclee2+cyclee3+t), color='g')
		t = np.add(t, cyclee4)
		plt.barh(pos, cyclee5, left=cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7+cyclee1+cyclee2+cyclee3+t, color='r')
	t = np.multiply(cyclee4, 35)
	plt.barh(pos, cyclee5, left=cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7+cyclee1+cyclee2+cyclee3+t, color='g')
	t = np.add(t, np.array([9,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
	plt.barh(pos, cyclee5, left=cycle_1+cycle_2+cycle_3+cycle_4+cycle_5+cycle_6
							   +cycle_7+cyclee1+cyclee2+cyclee3+t, color='r')
	# plt
	plt.yticks(pos, light_number)
	plt.show()

myexample()

def example():
	# library
	import matplotlib.pyplot as plt
	 
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