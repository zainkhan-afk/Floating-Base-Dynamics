import pygame
from Box2D import b2Filter
from utils import *

class FloatingBase:
	def __init__(self, sim_handle, position, angle, width, height, group_index):
		self.body = sim_handle.world.CreateDynamicBody(position=position, angle=angle)
		self.height = height
		self.width = width
		box = self.body.CreatePolygonFixture(box=(width, height), density=15, friction=0.3, filter = b2Filter(groupIndex=group_index))
		print(self.body.mass)
		
		print(self.body.position)

		self.color = (255, 0, 0)

	def Render(self, screen, PPM):
		H = screen.get_height()
		for fixture in self.body.fixtures:
			shape = fixture.shape
			vertices = [(self.body.transform * v) * PPM for v in shape.vertices]
			vertices = [(v[0], H - v[1]) for v in vertices]
			pygame.draw.polygon(screen, self.color, vertices)


	def GetTransform(self):
		return GetTransformationMatrix(self.body.angle, self.body.position[0], self.body.position[1])

	def GetPosition(self):
		return self.body.position