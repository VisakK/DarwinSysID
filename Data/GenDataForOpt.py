import numpy as np
import matplotlib.pyplot as plt
from Sim2Robot import *
import pickle

sit = np.loadtxt("states_conactions_SIT.txt")
stand = np.loadtxt("states_conactions_STAND.txt")
sitstand = np.loadtxt("states_conactions_SITSTAND.txt")
sitstand100 = np.loadtxt("states_conactions_SITSTAND100.txt")


sitstand_targetactions = np.loadtxt("GetUpTraj50.txt")
sitstand100_targetaction = np.loadtxt("GetUpTraj100.txt")


sit_sim = fromRobot(sit)
stand_sim = fromRobot(stand)
sitstand_sim = fromRobot(sitstand)
sitstand100_sim = fromRobot(sitstand100)

sitstand_a = fromRobot(sitstand_targetactions)
sitstand100_a = fromRobot(sitstand100_targetaction)


np.savetxt("sim_states_sit.txt",sit_sim,fmt="%1.5f")
np.savetxt("sim_states_stand.txt",stand_sim,fmt="%1.5f")
np.savetxt("sim_states_sitstand.txt",sitstand_sim,fmt="%1.5f")
np.savetxt("sim_states_sitstand100.txt",sitstand100_sim,fmt="%1.5f")

global_sit = np.array([0,0.32,0.,0.,0.,-0.44])
global_stand = np.array([0,0.22,0.,0.,0.,-0.37])

interpolated_global = np.zeros((50,6))

interpolated_global[:,1] = np.linspace(global_sit[1],global_stand[1],50)
interpolated_global[:,5] = np.linspace(global_sit[5],global_stand[5],50)


BatchedData = np.zeros((3,50,26))

BatchedData[0,:,:] = np.hstack((interpolated_global,sitstand_sim))
BatchedData[1,:,:] = np.hstack((np.tile(global_sit,(50,1)),sit_sim[:50,:]))
BatchedData[2,:,:] = np.hstack((np.tile(global_stand,(50,1)),stand_sim[:50,:]))

targetAction = np.zeros((3,50,26))
targetAction[0,:,:] = np.hstack((interpolated_global,sitstand_a))
targetAction[1,:,:] = np.hstack((np.tile(global_sit,(50,1)),sit_sim[:50,:]))
targetAction[2,:,:] = np.hstack((np.tile(global_stand,(50,1)),stand_sim[:50,:]))


with open('OptBatchData_hardwareResponse.pkl','wb') as fp:
	pickle.dump(BatchedData,fp)

with open('OptBatchData_targetActions.pkl','wb') as fp:
	pickle.dump(targetAction,fp)
