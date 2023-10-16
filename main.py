import sys
import numpy as  np

from simulation import Simulation
from ground import Ground
from Body import Body

from utils import *

PPM = 50.0 
TARGET_FPS = 60
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

sim = Simulation(width = SCREEN_WIDTH, height = SCREEN_HEIGHT, delta_T = TIME_STEP, PPM = PPM, FPS = TARGET_FPS)
ground = Ground(sim)
body = Body(sim, ground, position = np.array([5.0, 5.0]), angle  = 0)

sim.AddEntity(ground)
sim.AddEntity(body)

t = 0
while True:
	ret = sim.Step()
	t += TIME_STEP
	if not ret:
		sys.exit()