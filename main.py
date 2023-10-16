import sys
import numpy as  np

from simulation import Simulation
from ground import Ground
from Body import Body
from Controller import PID 
from utils import *

PPM = 50.0 
TARGET_FPS = 60
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

sim = Simulation(width = SCREEN_WIDTH, height = SCREEN_HEIGHT, delta_T = TIME_STEP, PPM = PPM, FPS = TARGET_FPS)
ground = Ground(sim)
body = Body(sim, ground, position = np.array([5.0, 5.0]), angle  = 0)
pid_controller = PID(body.dynamics_model, body.edge1_pos, body.edge2_pos, P = 50, I = 0.001, D = 10000)

sim.AddEntity(ground)
sim.AddEntity(body)

t = 0
ang = 0
new_state = None
x = 5.0
y = 6.0
while True:
	body.UpdateState()

	if new_state is not None:
		error_state = body.CalculateStateError(new_state)
		# error_state = current_state - new_state
		print("Difference between the predicted state and the actual state")
		print(error_state)

	current_state = body.GetState()
	J = body.GetJacobian()

	current_pos = current_state.position
	current_body_theta = current_state.body_theta

	goal_pos = np.array([x + 1*np.cos(ang), y + 1*np.sin(ang)])
	goal_pos = np.array([x, y])
	# goal_body_theta = current_body_theta
	# goal_pos = current_pos
	goal_body_theta = np.sin(ang)


	new_state, forces = pid_controller.Solve(current_state, J, current_pos, current_body_theta, goal_pos, goal_body_theta)
	# forces = np.array([0, 1, 0, 1]).astype("float")
	# print(forces.shape)
	body.ApplyState(new_state, forces)

	# print()
	# print(current_state)
	# print(new_state)

	ret = sim.Step()
	t += TIME_STEP
	ang += 0.001
	if not ret:
		sys.exit()