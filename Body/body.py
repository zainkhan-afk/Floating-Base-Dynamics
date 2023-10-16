import pygame
import numpy as np
from utils import AlmostEqual, GetTransformationMatrix

from .floatingBase import FloatingBase
from .Dynamics import BodyDynamics
from .state import State

class Body:
	def __init__(self, sim_handle, ground, position = np.array([0, 0]), angle = 0):
		self.torso_width = 0.5
		self.torso_height = 0.1

		self.body_angle = angle

		self.floating_base = FloatingBase(sim_handle, position, angle, self.torso_width, self.torso_height, group_index = -1)

		self.state = State(position, np.array([0, 0]), np.array([0, 0]), angle, 0, 0)

		self.dynamics_model = self.SetUpBodyDynamics()

	def SetUpBodyDynamics(self):
		floating_body_pos = np.array([self.floating_base.body.position[0], self.floating_base.body.position[1]])

		dynamics = BodyDynamics(self.floating_base)

		return dynamics


	def UpdateState(self):
		hind_theta_thigh, hind_theta_shin = self.leg_hind.GetAngles()
		front_theta_thigh, front_theta_shin = self.leg_front.GetAngles()

		body_position = self.torso.GetPosition()
		body_angle = self.torso.body.angle

		body_position = np.array([body_position[0], body_position[1]])

		self.state = self.state.UpdateUsingJointTheta(np.array([hind_theta_thigh, hind_theta_shin, front_theta_thigh, front_theta_shin]))
		self.state = self.state.UpdateUsingPosition(body_position)
		self.state = self.state.UpdateUsingBodyTheta(body_angle)

	def GetState(self):
		return self.state

	def ApplyState(self, new_state):
		hind_leg_theta = new_state.joint_theta[:2]
		front_leg_theta = new_state.joint_theta[2:]

		self.leg_hind.SetAngles(hind_leg_theta)
		self.leg_front.SetAngles(front_leg_theta)

	def Render(self, screen, PPM):
		self.floating_base.Render(screen, PPM)

	def CalculateStateError(self, state_pred):
		error_state = self.state - state_pred
		return error_state