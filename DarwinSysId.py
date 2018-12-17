import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as fmin
from scipy.optimize import least_squares 
import cma
try:
	import pydart2 as pydart
	pydart.init()
except ImportError:
	print("Install Pydart2")
import pickle
# This script takes a motion sequence from hardware and simulation and 
# optimizes for certain predefined parameters. For example, PID gains, damping, friction
# How optimization works :
# Data is separated into chunks of 5 seconds
# Set initial state -> Rollout for 5 seconds -> compute L1 and L2 norm of difference in Hardware and 
# simulation.


class DarwinEnv():
	def __init__(self,dt=0.002,control_frequency=20,optimizeMu=False,
		TorqueLimits=2.5,includeIMU=False,rollout_length=2.0,):
		# Initialize Pydart2 and other parameters here
		# This only does DARWIN-OP2 for now
		self.dt = dt
		self.control_frequency = control_frequency

		self.optimizeMu = optimizeMu
		self.TorqueLimits = TorqueLimits
		self.includeIMU = includeIMU
		self.rollout_length = rollout_length
		
		self.DarwinWorld = pydart.World(self.dt)
		self.DarwinWorld.add_skeleton('ground1.urdf')
		self.DarwinWorld.add_skeleton('robotis_op2.urdf')

		self.skel = self.DarwinWorld.skeletons[-1] # last skel
		self.DarwinWorld.set_gravity([0.,0.,-9.81])
		self.ndofs = self.skel.ndofs
		self.preverror = np.zeros(self.ndofs,)
		self.edot = np.zeros(self.ndofs,)
		self.error = []
		for i in range(self.skel.ndofs):
			j = self.skel.dof(i)
			j.set_damping_coefficient(0.275)
		
		# Enforce Joint Limits
		for joint in range(0,len(self.skel.joints)):
			for dof in range(len(self.skel.joints[joint].dofs)):
				if self.skel.joints[joint].has_position_limit(dof):
					self.skel.joints[joint].set_position_limit_enforced(True)

		for body in self.skel.bodynodes:
			if body.name == "base_link":
				body.set_mass(0.001)
			if body.name == "MP_PMDCAMBOARD":
				body.set_mass(0.001)

	def SetPIDgains(self,gains):
		self.kp = gains[0]
		self.kd = gains[1]



	def SetMu(self,mu):
		# mu - {damping,stiffness,friction}
		for i in range(self.skel.ndofs):
			j = self.skel.dof(i)
			j.set_damping_coefficient(mu[0])
			j.set_spring_stiffness(mu[1])

		for body in self.skel.bodynodes + self.DarwinWorld.skeletons[0].bodynodes:
			body.set_friction_coeff(mu[2])



	def PID(self,):

		q = self.skel.q
		dq = self.skel.dq
		tau = np.zeros(self.ndofs)
		for i in range(6,self.ndofs):
			tau[i] = -self.kp[i - 6] * \
						(q[i] - self.target[i]) - \
							self.kd[i - 6] *dq[i]

		torques = self.ClampTorques(tau)

		return torques

	def ClampTorques(self,torques):

		for i in range(6,self.ndofs):
			if torques[i] > self.TorqueLimits:#
				torques[i] = self.TorqueLimits 
			if torques[i] < -self.TorqueLimits:
				torques[i] = -self.TorqueLimits

		return torques


	def advance(self,):


		Tau = np.zeros(self.ndofs)
		for i in range(int((1/self.control_frequency)/self.dt)):
			Tau = self.PID()
			#print("tau",Tau)

			self.skel.set_forces(Tau)
			self.DarwinWorld.step()





	def step(self,targets):

		self.target = targets
		self.advance()

		currentPosition = self.skel.q
		currentVelocity = self.skel.dq

		return currentPosition,currentVelocity


	def FirstState(self,pose):
		self.DarwinWorld.reset()
		self.skel.set_positions(pose)
		self.skel.set_velocities(np.zeros(self.ndofs,))



	def RollOut(self,targetPoses):

		t = 0
		i = 0
		

		self.FirstState(targetPoses[0,:])
		p = []
		v = []
		while t < self.rollout_length:
			pos,vel = self.step(targetPoses[i,:])

			p.append(pos.tolist())
			v.append(vel.tolist())
			
			t += 1/self.control_frequency
			i+=1


		return np.asarray(p),np.asarray(v)




	def Optimizer(self,x,targets,RealPoses):

		# Set PID 45,3.1,0.2575,0.50,10.5 55.,3.5,0.5775,0.75,12.65,
		a = 45 + 10*x[0] #54.19
		b = 2.5 + x[1] #2.5002
		c = 0.2575 + 0.12*x[2] #0.26498
		d = 0.50 + 0.25*x[3] #0.6925
		e = 10.5 + 2*x[4] #12.3018
		
		if not self.optimizeMu:
			self.SetPIDgains(x.reshape((2,20)))
		# Set Mu
		if self.optimizeMu:
			gains = [[a]*20,[b]*20]
			self.SetPIDgains(gains)
			Mu = [c,d,e]
			self.SetMu(Mu)
		# Begin Rollout of 5 Seconds.
		SimPos,SimVel = self.RollOut(targets)

		
		# COmpute Error
		e = np.sum(np.sum(np.abs(SimPos - RealPoses),axis=1)) + np.sum(np.sum((SimPos-RealPoses)**2,axis=1))

		#print("eror",e)
		#self.error.append(e)
		return e

	
	def SysID(self,):

		# Load Batched Dataset here - size - [n*100*26]
		TestPoses = np.zeros(26,)
		TestPoses[5] = -0.35

		with open("./Data/OptBatchData_hardwareResponse.pkl","rb") as fp:
			HR = pickle.load(fp)

		with open("./Data/OptBatchData_targetActions.pkl","rb") as fp:
			TA = pickle.load(fp)

		print(HR.shape)
		print(TA.shape)

		TestPoses = np.tile(TestPoses,(101,1))
		#gains0 = np.array([[2.1,1.79,4.93,2.0,2.02,1.98,2.2,2.06,148,152,150,136,153,102,151,151.4,150.45,151.36,154,105.2],
		#[0.21,0.23,0.22,0.25,0.21,0.26,0.28,0.213,0.192,0.198,0.22,0.199,0.02,0.01,0.53,0.27,0.21,0.205,0.022,0.056]])

		#error = self.Optimizer(gains0,TestPoses,TestPoses)45,3.1,0.2575,0.50,10.5 55.,3.5,0.5775,0.75,12.65,
		


		#res = fmin(self.Optimizer,gains0.flatten(),method='Nelder-Mead',args=(TestPoses,TestPoses),options={'xtol':1e-8,'disp':True})
		X0 = np.array([50,3.2,0.275,0.5,10.5])
		X0 = np.array([0.5,0.5,0.5,0.5,0.5])

		for i in range(HR.shape[0]):
			es = cma.CMAEvolutionStrategy(X0,0.5,{'verb_disp':1,'maxfevals':1e4,'bounds':[[0,0,0,0,0],[1,1,1,1,1]]})
			res = es.optimize(self.Optimizer,args=(TA[i,:,:],HR[i,:,:])).result()
			
			
			X0 = res[0]
			
			with open("CurrentResult.txt","wb") as fp:
				np.savetxt(fp,np.asarray(X0),fmt="%1.6f")
		#print("Result",X0)
		#print(es.result_pretty())

		#plt.plot(self.error)
		#plt.show()
		#print("Result :",res.x) 




		# In a loop select a batch do optimization and repeat for all the Train dataset


	def TestOptimization(self,):
		return None



if __name__ == "__main__":
	env = DarwinEnv(optimizeMu=True,rollout_length=2.49)
	env.SysID()








