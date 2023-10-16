import numpy as np

from .floatingBase import FloatingBase

from utils import GetInverseMatrix, GetAngle

class BodyDynamics:
	def __init__(self, floating_base):
		self.floating_base = FloatingBase(floating_base)
		self.links = []
		self.links.append(self.floating_base)
		self.composite_inertia = None

	def CalculateCompositeRigidBodyInertiaWRTWorld(self):
		self.composite_inertia = np.zeros((3, 3))
		
		for link in self.links:
			world_T_link = link.GetTransform()
			self.composite_inertia += world_T_link@link.GetSpatialInertia()@(world_T_link.T)

	def CalculateCompositeRigidBodyInertiaWRTFloatingBase(self):
		world_T_FB = self.links[0].GetTransform()
		FB_T_world = GetInverseMatrix(world_T_FB)
		
		self.composite_inertia = np.zeros((3, 3))
		
		for link in self.links:
			world_T_link = link.GetTransform()
			FB_T_link = FB_T_world@world_T_link
			self.composite_inertia += FB_T_link@link.GetSpatialInertia()@(FB_T_link.T)

	def GetCompositeInertia(self):
		return self.composite_inertia

	def ForwardDynamics(self, forces, jacobian, state):
		M = self.GetMassMatrix(state)
		C = self.GetCoriolisMatrix(state)
		G = self.GetGravityMatrix(state)

		EE_hind = np.array([[jacobian[2, 0], jacobian[2, 1]]])
		EE_front  = np.array([[jacobian[2, 2], jacobian[2, 3]]])
		
		f_hind = np.array([[forces[0, 0], forces[1, 0]]])
		f_front  = np.array([[forces[2, 0], forces[3, 0]]])

		force_angle_hind  = GetAngle(f_hind.ravel(), np.array([0, 0]))
		force_angle_front = GetAngle(f_front.ravel(), np.array([0, 0]))

		torque_from_hind = np.cross(EE_hind, f_hind)[0]
		torque_from_front = np.cross(EE_front, f_front)[0]

		forces_body = jacobian@forces

		resultant_torques = np.zeros((3, 1))
		resultant_torques[:3, :] = forces_body
		resultant_torques[2, 0] = torque_from_front + torque_from_hind

		M_inv = GetInverseMatrix(M)

		# print(forces)
		print(resultant_torques)
		print(G)
		print(resultant_torques - C - G)

		theta_double_dot = M_inv@(resultant_torques - C - G)

		return theta_double_dot


	def GetMassMatrix(self, state):
		M = self.composite_inertia
		return M

	def GetCoriolisMatrix(self, state):
		C = np.zeros((3, 1))
		return C

	def GetGravityMatrix(self, state):
		G = self.floating_base.GetGravityMatrix()
		return G