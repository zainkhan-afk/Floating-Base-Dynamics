import pygame
import numpy as np
from utils import AlmostEqual, GetTransformationMatrix

from .floatingBase import FloatingBase
from .Dynamics import BodyDynamics
from .state import State
from .kinematics import BodyKinematics

class Body:
	def __init__(self, sim_handle, ground, position = np.array([0, 0]), angle = 0):
		self.torso_width = 0.5
		self.torso_height = 0.1

		self.edge1_pos = (- self.torso_width, 0)
		self.edge2_pos = ( self.torso_width, 0)

		self.body_angle = angle

		self.floating_base = FloatingBase(sim_handle, position, angle, self.torso_width, self.torso_height, group_index = -1)

		self.state = State(position, np.array([0, 0]), np.array([0, 0]), angle, 0, 0)

		self.body_kine_model = BodyKinematics(self.edge1_pos, self.edge2_pos)

		self.dynamics_model = self.SetUpBodyDynamics()

	def SetUpBodyDynamics(self):
		floating_body_pos = np.array([self.floating_base.body.position[0], self.floating_base.body.position[1]])

		dynamics = BodyDynamics(self.floating_base)

		return dynamics


	def UpdateState(self):
		body_position = self.floating_base.GetPosition()
		body_angle = self.floating_base.body.angle

		body_position = np.array([body_position[0], body_position[1]])

		self.state = self.state.UpdateUsingPosition(body_position)
		self.state = self.state.UpdateUsingBodyTheta(body_angle)


	def GetJacobian(self):
		body_position = self.state.position
		body_theta = self.state.body_theta

		COM_T_edge1 = GetTransformationMatrix(0, self.edge1_pos[0], self.edge1_pos[1])
		COM_T_edge2 = GetTransformationMatrix(0, self.edge2_pos[0], self.edge2_pos[1])

		COM_T_edge1_rotated = GetTransformationMatrix(self.state.body_theta, 0, 0)@COM_T_edge1
		COM_T_edge2_rotated = GetTransformationMatrix(self.state.body_theta, 0, 0)@COM_T_edge2

		world_T_COM = GetTransformationMatrix(self.state.body_theta, self.state.position[0], self.state.position[1])

		world_T_edge1 = world_T_COM@COM_T_edge1
		world_T_edge2 = world_T_COM@COM_T_edge2

		edge1_pos_wrt_world = np.array([[world_T_edge1[0, -1], world_T_edge1[1, -1]]]).T
		edge2_pos_wrt_world = np.array([[world_T_edge2[0, -1], world_T_edge2[1, -1]]]).T

		edge1_pos_wrt_FB = np.array([[COM_T_edge1_rotated[0, -1], COM_T_edge1_rotated[1, -1]]]).T
		edge2_pos_wrt_FB = np.array([[COM_T_edge2_rotated[0, -1], COM_T_edge2_rotated[1, -1]]]).T

		# body_J  = self.body_kine_model.GetJacobian(edge1_pos_wrt_world, edge2_pos_wrt_world)
		body_J  = self.body_kine_model.GetJacobian(edge1_pos_wrt_FB, edge2_pos_wrt_FB)

		return body_J


	def GetState(self):
		return self.state

	def ApplyState(self, new_state, forces):
		COM_T_edge1 = GetTransformationMatrix(0, self.edge1_pos[0], self.edge1_pos[1])
		COM_T_edge2 = GetTransformationMatrix(0, self.edge2_pos[0], self.edge2_pos[1])

		COM_T_edge1_rotated = GetTransformationMatrix(self.state.body_theta, 0, 0)@COM_T_edge1
		COM_T_edge2_rotated = GetTransformationMatrix(self.state.body_theta, 0, 0)@COM_T_edge2

		world_T_COM = GetTransformationMatrix(self.state.body_theta, self.state.position[0], self.state.position[1])

		world_T_edge1 = world_T_COM@COM_T_edge1
		world_T_edge2 = world_T_COM@COM_T_edge2

		edge1_pos_wrt_world = np.array([world_T_edge1[0, -1], world_T_edge1[1, -1]])
		edge2_pos_wrt_world = np.array([world_T_edge2[0, -1], world_T_edge2[1, -1]])

		# self.floating_base.body.ApplyForce(force=(forces[2], forces[3]), point=(self.state.position[0], self.state.position[1]), wake=True)

		self.floating_base.body.ApplyForce(force=(forces[0], forces[1]), point=(edge1_pos_wrt_world[0], edge1_pos_wrt_world[1]), wake=True)
		self.floating_base.body.ApplyForce(force=(forces[2], forces[3]), point=(edge2_pos_wrt_world[0], edge2_pos_wrt_world[1]), wake=True)

	def Render(self, screen, PPM):
		self.floating_base.Render(screen, PPM)

	def CalculateStateError(self, state_pred):
		error_state = self.state - state_pred
		return error_state