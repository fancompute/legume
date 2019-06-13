import numpy as np

def init_lattice(a1=np.array([1, 0]), a2=np.array([0, 1])):
	'''
	Initialize a Bravais lattice; default is square
	'''
	ec_area = np.linalg.norm(np.cross(a1, a2))
	a3 = np.array([0, 0, 1])

	b1_3d = 2*np.pi*np.cross(a2, a3)[0:2]/np.dot(a1, np.cross(a2, a3)[0:2]) 
	b2_3d = 2*np.pi*np.cross(a3, a1)[0:2]/np.dot(a2, np.cross(a3, a1)[0:2])

	bz_area = np.linalg.norm(np.cross(b1_3d, b2_3d))

	lattice = {	'a1': a1,
				'a2': a2,
				'b1': b1_3d[0:2],
				'b2': b2_3d[0:2],
				'ec_area': ec_area,
				'bz_area': bz_area}

	return lattice