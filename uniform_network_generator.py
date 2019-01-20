import random
import sys
import numpy as np
import networkx as nx
import random
import scipy.sparse
import itertools
import sys
from scipy.sparse import csr_matrix
import networkx as nx

#TODO put it in the main
#define network parameters
# number of network participants (financial entities), e.g. banks
# density of the debpt network p
# number of reference entities

n_banks = 50

# notional values based on realisti values reported by Abaad et al.

max_notional = 50
mean_notional = 23



def is_naked(j, k, l, c):

	if (l[k][j] < naked_cds(j,k,c)):

		return 1
	else:
		return 0


def naked_cds(j,k,contract):
	'''
	This function is used to make a specific position either covered or naked
	It finds, given holder and reference entities, the sum of cds notional
	over all writers. 
	
	'''
	summ = 0
	for i in xrange(len(contract)):
		summ = summ + contract[i][j][k]
	return summ

def uniform_network_gen (n_banks, p):
	'''
    Return the uniform network generated given denisty (p), share of naked CDSs : 55-60 %
    
	type:
    
    debt_matrix, 2d-array of debt contracts
    contract,  3d-array of cds contracts
    ref_entities, list of reference entities


	Parameters:

    n_banks: int (number of banks, nodes, in the network)
    p : float ( density of network should be in (0, 1))

	'''

	ref_entities = []
	n_ref_entities = n_banks/3
	n_c_d = 9
	
	naked_cds_share = 0.6

	size = n_ref_entities
	while size > 0:
		
		ref_entity = random.randint(0, n_ref_entities-1)

		if  ref_entity not in ref_entities: #// pathalogical cases are ommited
			ref_entities.append(ref_entity)
			size -= 1

	# generate debt network
	# erdos_rany game : G(n, m)
	# m = (n(n-1))/2*p

	m = n_banks*(n_banks-1)/2*p
	
	G = nx.gnm_random_graph(n_banks, m, seed=3, directed=True)
	
	debt_matrix = [[0 for i in range(n_banks)] for j in range(n_banks)]
	
	A = nx.adjacency_matrix(G) # type A : scipy sparce scr matrix

	cx = A.tocoo() 

	for i in range(n_banks):
		for j in range(n_banks):
		
			if i in ref_entities:
				
				debt_matrix[i][j] = random.randint(1,mean_notional-5)
			else :
				debt_matrix[i][j] = random.randint(1,max_notional-10)


	contract = [[[0 for i in range(n_banks)] for j in range(n_banks)] for k in range(n_banks)]

	for k in ref_entities:
		i = 0
		j = 0
		#TODO Define number of contracts written such that it is more acurate and realistic
		n_contracts_written = random.randint(1,n_banks/n_c_d)

		while n_contracts_written>0:
			
			while i==j or i == k or j == k  :
				i = random.randint(0,n_banks-1)
				j = random.randint(0,n_banks-1)
			if contract[i][j][k] ==0 :#and has_liability(k, liability)  :

				contract[i][j][k] = random.randint(1, mean_notional)

				n_contracts_written -=1
	
	
			i = 0
			j = 0	
	
	return debt_matrix, contract, ref_entities


#TO CHECK SHARE FO NAKED CDSs UNCOMMENT FOLLOWINGS.

# l,c,ref = uniform_network_gen(50, 0.3)
# n=0
# m=0
# counter = 0
# vec= []
# for x in range(10000):

# 	for i in range(50):
# 		for j in range(50):
# 			for k in ref:

# 				if c[i][j][k] !=0 :
					
# 					n+=is_naked(j,k,l,c)
# 					counter+=1

# 	r= float(n)/counter
# 	vec.append(r)

# print np.average(vec)