from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import pygame
class Ground:
	def __init__(self, sim_handle):
		self.body = sim_handle.world.CreateStaticBody(
						    position=(0, 0.5),
						    shapes=polygonShape(box=(13, 0.5)),
						)

		self.color = (125, 125, 125)


	def Render(self, screen, PPM):
		H = screen.get_height()
		for fixture in self.body.fixtures:
			shape = fixture.shape
			vertices = [(self.body.transform * v) * PPM for v in shape.vertices]
			vertices = [(v[0], H - v[1]) for v in vertices]
			pygame.draw.polygon(screen, self.color, vertices)