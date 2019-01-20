import networkx as nx
import random
import sys
import networkx as nx
import random
import scipy.sparse
import itertools
import sys
from scipy.sparse import csr_matrix
import numpy as np
import math
import network_measures as nm


def net_position (i, debt_matrix):

	net_pos = sum(debt_matrix[i][:]-debt_matrix[:][i])

	return net_pos


def covered_cds(j,k,contract):
	'''
	This function is used to make a specific position either covered or naked
	It finds, given holder and reference entities, the sum of cds notional
	over all writers. 
	'''
	summ = 0
	for i in xrange(len(contract)):
		summ = summ + contract[i][j][k]
	return summ

def bowtie_network_gen(n_banks, net_buyers_fraction, net_sellers_fraction, density_overall, prob_dealer_dealer):
	'''

  	Return the bow tie network generated given parameters, share of naked CDSs 55-60%
    
	type : 
	    
	    debt_matrix, 2d-array of debt contracts
	    contract,  3d-array of cds contracts
	    ref_entities, list of reference entities


	Parameters:

	    n_banks: int (number of banks, nodes, in the network)
	    net_buyers_fraction : float ( share of buyers in the network)
	    net_sellers_fraction : float ( share of sellers in the network)
	    density_overall : float ( density of the core)
	    prob_dealer_dealer : float ( probability of link formation among dealers)



	'''


	ref_entities = []
	n_ref_entities = n_banks/3
	size = n_ref_entities
	while size > 0:
			
		ref_entity = random.randint(0, n_ref_entities-1)

		if  ref_entity not in ref_entities: #// pathalogical cases are ommited
			ref_entities.append(ref_entity)
			size -= 1


	n_net_cds_buyers = int(math.floor(n_banks * net_buyers_fraction))
	n_net_cds_sellers = int(math.floor(n_banks * net_sellers_fraction))
	n_net_buyers_and_sellers = int(math.floor(n_net_cds_sellers+n_net_cds_buyers))
	n_dealers =int( n_banks- n_net_buyers_and_sellers)
	pdealer = prob_dealer_dealer
	debt_matrix_of_dealers = [[0 for i in range(n_dealers)] for j in range(n_dealers)]
	debt_matrix = [[0 for i in range(n_banks)] for j in range(n_banks)]
	
	


	G_dealers = nx.newman_watts_strogatz_graph(n_banks, 3, 0.5, seed=None)
	A = nx.adjacency_matrix(G_dealers) # type A : scipy sparce scr matrix
	# print A
	cx = A.tocoo()	
	for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
		if random.randint(0,1):
			debt_matrix[i][j] =1
		else :
			debt_matrix[j][i]=1	
	n_edges =  np.sum(np.asarray(debt_matrix))		
	n_edges_wout_dealers = density_overall*(n_banks-1)*(n_banks-1)-n_edges
	p_bs_to_dealer = n_edges_wout_dealers / (n_dealers*(net_buyers_fraction+net_sellers_fraction))

	if (p_bs_to_dealer<0):
		p_bs_to_dealer = -0.00001
	
	prob_dealer_buyer = p_bs_to_dealer
	prob_dealer_seller = p_bs_to_dealer
	
	if prob_dealer_seller>1:
		prob_dealer_seller =1

 	if prob_dealer_buyer>1 :
 		prob_dealer_buyer = 1
	print n_dealers
	print prob_dealer_buyer 	
 	for i in range(n_net_cds_buyers):
		n_buyer_dealer = np.random.binomial(n_dealers, prob_dealer_buyer)
		buyer_index = n_dealers+i
  		
		for j in range(n_buyer_dealer):
			dealer_index = random.randint(0,n_dealers-1)
			debt_matrix[dealer_index][buyer_index] = 1


  	for i in range(n_net_cds_sellers):
  		n_seller_dealer = np.random.binomial(n_dealers, prob_dealer_seller)
  		seller_index = n_net_cds_buyers+n_dealers+i

	  	for j in range(n_seller_dealer):
	  		dealers_index = random.randint(0,n_dealers-1)
	  		debt_matrix[dealer_index][seller_index] = 1


	for i in range(n_banks):
		for j in range(n_banks):

			if i in ref_entities:

			
				debt_matrix[i][j] = np.random.binomial(50,0.2)#+covered*random.randint(0,1)

			elif i in range(n_dealers):
				debt_matrix[i][j] = np.random.binomial(50,0.35)
		



	contract = [[[0 for i in range(n_banks)] for j in range(n_banks)] for k in range(n_banks)]

	for k in ref_entities:
		i = 0
		j = 0
		n_contracts_written = np.random.binomial(20,0.3)

		while n_contracts_written>0:
			
			while i==j or i == k or j == k  :
				
				# If the chosen writer is a dealer then the holder should be chosen from ultimate CDS sellers,
				# conversely, if it is a CDS seller then the holder needs to be a dealer.
				
				which = random.randint(0,1)
				
				i = random.randint(n_dealers+n_net_cds_buyers,n_dealers+n_net_cds_sellers+n_net_cds_buyers-1)*which+random.randint(0,n_dealers-1)*(1-which)
				
				if which ==1:
					j = random.randint(0,n_dealers-1)
				else :
					j = random.randint(n_dealers,n_net_cds_buyers+n_dealers)	

			if contract[i][j][k] ==0 :#and has_liability(k, liability)  :

				contract[i][j][k] = np.random.binomial(50,0.2) # * random.randint(0,1)

				n_contracts_written -=1
	
	
			i = 0
			j = 0	

	debt_csr_matrix=csr_matrix(debt_matrix)
	cx = debt_csr_matrix.tocoo() 


	return debt_matrix, contract, ref_entities	
	


# l,c,ref= bowtie_network_gen(50, 0.2, 0.4, 0.1, 0.6)

# n=0
# counter = 0
# vec= []
# for x in range(1000):

# 	for i in range(50):
# 		for j in range(50):
# 			for k in ref:

# 				if c[i][j][k] !=0 :
					
# 					n+=nm.is_naked(j,k,l,c)
# 					counter+=1

# 	r= float(n)/counter
# 	vec.append(r)

# print np.average(vec)