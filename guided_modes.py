''' 
Function to compute the guided modes of a multi-layer structure
Input
	g_array  		: numpy array of wave vector amplitudes 
	eps_array		: numpy array of slab permittivities, starting with lower 
					  cladding and ending with upper cladding
	n_modes			: maximum number of solutions to look for, starting from 
					  the lowest-frequency one
Output
	om_guided   	: array of size n_modes x length(g_array) with the guided 
					  mode frequencies
	(Will need further outputs in the future)  
'''

def guided_modes(g_array, eps_array, n_modes=1):
 	
 	return om_guided 