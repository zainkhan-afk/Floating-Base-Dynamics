import numpy as np

class SimulateDynamics:
	def __init__(self, robotDynamicsModel):
		self.robotDynamicsModel = robotDynamicsModel

	def GoToNextStateFD(self, force, J, current_state):
		# self.robotDynamicsModel.CalculateCompositeRigidBodyInertiaWRTWorld()
		self.robotDynamicsModel.CalculateCompositeRigidBodyInertiaWRTFloatingBase()
		
		theta_double_dot, self.old_torques = self.robotDynamicsModel.ForwardDynamics(force, J, current_state)

		joint_theta_double_dot = theta_double_dot[3:, 0]
		body_acc = theta_double_dot[:2, 0]
		body_theta_double_dot = theta_double_dot[2, 0]

		new_state = current_state.UpdateUsingJointThetaDoubleDot(joint_theta_double_dot)
		new_state = new_state.UpdateUsingAcceleration(body_acc)
		new_state = new_state.UpdateUsingBodyThetaDoubleDot(body_theta_double_dot)

		return new_state