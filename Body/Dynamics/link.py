import numpy as np

class Link:
	def __init__(self, link_object):
		self.link_object = link_object
		self.I = None

		self.CalculateSpatialInertia()

	def CalculateSpatialInertia(self):
		self.Inertia = 1/12 * self.link_object.body.mass * (self.link_object.width**2 + self.link_object.height**2)

		self.I = np.array([
						[self.link_object.body.mass,   						  0,  							  0],
						[  		 				  0, self.link_object.body.mass,  							  0],
						[ 		 				  0,   						  0,  self.link_object.body.inertia]
						])


	def GetSpatialInertia(self):
		return self.I


	def GetTransform(self):
		return self.link_object.GetTransform()