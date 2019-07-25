import numpy as np
from .utils import toeplitz_block

from autograd.extend import primitive, defvjp

''' Define here various primitives needed for the main code 
To use with both numpy and autograd backends, define the autograd primitive of 
a numpy function fnc as fnc_ag, and then define the vjp'''

toeplitz_block_ag = primitive(toeplitz_block)

def vjp_maker_TB_T1(Tmat, n, T1, T2):
	''' Gives vjp for Tmat = toeplitz_block(n, T1, T2) w.r.t. T1'''
	def vjp(v):
		ntot = Tmat.shape[0]
		p = int(ntot/n) # Linear size of each block
		vjac = np.zeros(T1.shape, dtype=np.complex128)

		for ind1 in range(n):
			for ind2 in range(ind1, n):
				for indp in range(p):
					vjac[(ind2-ind1)*p:(ind2-ind1+1)*p-indp] += \
						v[ind1*p + indp, ind2*p+indp:(ind2+1)*p]

					if ind2 > ind1:
						vjac[(ind2-ind1)*p:(ind2-ind1+1)*p-indp] += \
							np.conj(v[ind2*p+indp:(ind2+1)*p, ind1*p + indp])
		return vjac

	return vjp

def vjp_maker_TB_T2(Tmat, n, T1, T2):
	''' Gives vjp for Tmat = toeplitz_block(n, T1, T2) w.r.t. T2'''
	def vjp(v):
		ntot = Tmat.shape[0]
		p = int(ntot/n) # Linear size of each block
		vjac = np.zeros(T2.shape, dtype=np.complex128)

		for ind1 in range(n):
			for ind2 in range(ind1, n):
				for indp in range(p):
					vjac[(ind2-ind1)*p+1:(ind2-ind1+1)*p-indp] += \
						v[ind1*p+indp+1:(ind1+1)*p, ind2*p+indp]

					if ind2 > ind1:
						vjac[(ind2-ind1)*p+1:(ind2-ind1+1)*p-indp] += \
							np.conj(v[ind2*p+indp, ind1*p+indp+1:(ind1+1)*p])
		return vjac

	return vjp

defvjp(toeplitz_block_ag, None, vjp_maker_TB_T1, vjp_maker_TB_T2)