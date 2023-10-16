import numpy as np
from Body.Dynamics.simulateDynamics import SimulateDynamics
from utils import GetTransformationMatrix

class PID:
	def __init__(self, dynamicsModel, edge1_pos, edge2_pos, P = 300, I = 0, D = 50):
		self.dynamicsSimulator = SimulateDynamics(dynamicsModel)
		self.edge1_pos, self.edge2_pos = edge1_pos, edge2_pos
		self.P = P;
		self.I = I;
		self.D = D;
  
		self.prev_error = np.array([0, 0, 0, 0]).astype("float32")
		self.error_sum = np.array([0, 0, 0, 0]).astype("float32")

		self.prev_goal_pos = None

	def GetEdgePosition(self, position, theta, edge_position):
		world_T_body = GetTransformationMatrix(0, position[0], position[1])
		body_T_edge   = GetTransformationMatrix(theta, 0, 0)

		edge_position_world = world_T_body@body_T_edge@edge_position.T

		return [edge_position_world[0, 0], edge_position_world[1, 0]]


	def Solve(self, current_state, J, current_pos, current_body_theta, goal_pos, goal_body_theta):
		edge1_position_current  = self.GetEdgePosition(current_pos, current_body_theta,
									edge_position = np.array([[self.edge1_pos[0], self.edge1_pos[1], 1]]))
		edge2_position_current = self.GetEdgePosition(current_pos,  current_body_theta,
									edge_position = np.array([[self.edge2_pos[0], self.edge2_pos[1], 1]]))

		edge1_position_goal  = self.GetEdgePosition(goal_pos, goal_body_theta,
									edge_position = np.array([[self.edge1_pos[0], self.edge1_pos[1], 1]]))
		edge2_position_goal = self.GetEdgePosition(goal_pos,  goal_body_theta,
									edge_position = np.array([[self.edge2_pos[0], self.edge2_pos[1], 1]]))

		current_pos = np.array(edge1_position_current + edge2_position_current)
		goal_pos    = np.array(edge1_position_goal + edge2_position_goal)

		error = goal_pos - current_pos
		force = self.P*error + self.D*(error - self.prev_error) + self.I*self.error_sum

		new_state = self.dynamicsSimulator.GoToNextStateFD(force.reshape(len(force), 1), J, current_state)

		self.prev_error = error
		self.error_sum += error

		return new_state, force