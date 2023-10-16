from renderer import Renderer

import pygame
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
from utils import *


class Simulation:
	def __init__(self, width = 640, height = 480, delta_T = 0.1, PPM = 20, FPS = 60):
		self.entities = []
		
		self.width = width
		self.height = height
		self.delta_T = delta_T
		self.PPM = PPM
		self.FPS = FPS

		self.clock = pygame.time.Clock()
		self.renderer = Renderer(width, height)
		self.world = world(gravity=(0, gravity), doSleep=True)

	def AddEntity(self, entity):
		self.entities.append(entity)

	def Render(self):
		simulation_running = self.renderer.Poll()

		if simulation_running:
			self.renderer.Clear()
			self.renderer.Render(self.entities)


		return simulation_running

	def Step(self):
		simulation_running = self.renderer.Poll()

		if simulation_running:
			self.renderer.Clear()
			self.renderer.Render(self.entities, self.PPM)

			self.world.Step(self.delta_T, 10, 10)
			# self.clock.tick(self.FPS)
		return simulation_running