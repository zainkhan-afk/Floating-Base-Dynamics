from .link import Link
import numpy as np
from utils import *

class FloatingBase(Link):
	def __init__(self, link_object):
		super().__init__(link_object)

	def GetGravityMatrix(self):
		G = np.array([
						[0],
						[-self.link_object.body.mass*gravity],
						[0]
					 ])

		return G