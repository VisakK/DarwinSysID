import numpy as np

# These functions convert the joint angles between 

def fromRobot(positions):
	# reorder joints
	order=[1,3,5,0,2,4,18,19,7,9,11,13,15,17,6,8,10,12,14,16]
	# convert from int values to radians
	simState = np.zeros(positions.shape)
	simState = (positions - 2048)*(np.pi/180)*0.088
	



	return simState[:,order]

def toRobot(positions):
	# reorder joints
	order = [3,0,4,1,5,2,14,8,15,9,16,10,17,11,18,12,19,13,6,7]
	
	robotState = np.zeros(positions.shape)
	
	robotState = int(positions*180*(1/(np.pi*0.088))) + 2048

	return robotState[:,order]


