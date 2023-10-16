import pygame
import sys

class Renderer:
	def __init__(self, width, height):
		self.background_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((width, height))
		pygame.display.set_caption("Rectangle Drawing")

	def Poll(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				
				return False

		return True

	def Clear(self):
		# Clear the screen with white color
		self.screen.fill(self.background_color)

	def Render(self, entities, PPM):
		for entity in entities:
			entity.Render(self.screen, PPM)
		
		pygame.display.flip()

	def DrawPolygon(self, color, vertices):
		pygame.draw.polygon(self.screen, color, vertices)
		pygame.display.flip()