import networkx as nx
import graph_tool as graph_tool
from graph_tool import *
from graph_tool import centrality
from graph_tool import clustering
from graph_tool import topology
from graph_tool import correlations
import scipy as scipy
from scipy import stats
import numpy as np


def naked_cds(j,k,contract):
  

  """Return sum of the value of notionals writen on k in which j is holder.


    type : float

    Parameters
    ----------
    j : int  ( index of j)
    i : int (index of i)
    contract : 3d-array (matrix of CDS contracts) 
    
  """
  summ = 0

  for i in xrange(len(contract)):
    summ = summ + contract[i][j][k]

  return summ	

def is_naked(j, k, l, c):

  """Return 1 if j has naked position towards k; otherwise return 0


  type : int

  Parameters
  ----------
  j : int  ( index of j)
  k : int (index of i)
  l : 2d-array ((debt contracts)liability matrix)
  contract : 3d-array (matrix of CDS contracts) 

  """

  if (l[k][j] < naked_cds(j,k,c)):
    return 1
  else:
    return 0
def cds_over_debt(j, k, l, c):

  """Return Return cds over debt ratio


  type : float

  Parameters
  ----------
  j : int  ( index of j)
  k : int (index of i)
  l : 2d-array ((debt contracts)liability matrix)
  contract : 3d-array (matrix of CDS contracts) 

  """
  a = naked_cds(j,k,c)/(l[k][j]+0.00000000000000000000000000000001)
  
	
  return a
		
	

def convert_to_networkx_object(net):

  """Return the graph object associated with the adjacency matrix of a network


  type : networkx Graph

  Parameters
  ----------
  net : 2d array  ( adjacency matrix (weighted or non-weighted))
  
  """

  g = nx.from_numpy_matrix(np.asarray(net), create_using=nx.DiGraph())


  return g

def convert_to_graphtool_object(net):
  
  """Return the graph object associated with the adjacency matrix of a network,as well
    as its corresponding weights


  type : graph-tool Graph
  		 2d array 	

  Parameters
  ----------
  net : 2d array  ( adjacency matrix (weighted or non-weighted))
  
  """

  g = Graph(directed=True)
  g.add_vertex(len(net))
  edge_weights = g.new_edge_property('double')
  for i in range(len(net)):
    for j in range(len(net)):
      if i!=j and net[i][j]!=0:
        e = g.add_edge(i, j)
        edge_weights[e] = net[i][j]
  return g, edge_weights    


def convert_to_sep_networks(li, c, ref_entities):

  """Return different subnetworks, i.e., CDS and debt subnetwork; and the overall network
   +
   number of naked CDSs  (int)
   number of all contracts (int )
   number of red only cycles (int )
   average, std, and maximum value of cds to debt ratio
   
  Parameters
  ----------

  li : 2d array (liability matrix, debt contract)   
  c : 3d array (cds contracts)
  ref_entities : list (set of reference entities)

  """



  n_banks = len(li)

  # define this parameter, weight of dependency edges in the colored dependency graph,
  # edges steming from ref entity to both holder and writer of contract
  
  cds_indirect_weight = 0.01
  
  debt_network = [[0 for i in range(n_banks)] for j in range(n_banks)]
  cds_network = [[0 for i in range(n_banks)] for j in range(n_banks)]
  
  naked_cds_network = cds_network
  
  aggregated_network = [[0 for i in range(n_banks)] for j in range(n_banks)]
  
  weighted_aggregated_network = aggregated_network
 

  
#generate matrices of debt, and cds subnetwork
  
  num_naked_cds = 0
  
  tot_cont = 0
  
  tot_debt=0
  
  cds_over_debts = []
  
  for i in range(n_banks):
  
    for j in range(n_banks):

      if li[i][j] != 0 :
      	tot_debt +=1
        debt_network[i][j] = li[i][j]

      for k in ref_entities:
        if c[i][j][k] !=0:	

          if li[k][j]>0:
            
            # calculate cds over debts ratio; measure of nakedness

            cds_over_debts.append(cds_over_debt(j,k,li,c))
          

          if is_naked(j,k,li,c):

            naked_cds_network[k][j] = 1 

            num_naked_cds+=1
          
          tot_cont+=1

	
          cds_network[i][j] = c[i][j][k]
          cds_network[k][j] = cds_indirect_weight
          cds_network[k][i] = cds_indirect_weight

  '''
  naked cds network stores all the information of naked positions, then we'll calculate number of red only cycles given this network
  '''
  g,e = convert_to_graphtool_object(naked_cds_network)
  counter=0
  graph_tool.topology.all_circuits(g)
  num_elem_cycles = sum(1 for e in graph_tool.topology.all_circuits(g))	


  for i in range(n_banks):
    for j in range(n_banks):
    
      debt_edge = debt_network[i][j]
      cds_edge = cds_network[i][j]
      
      if debt_edge !=0 or cds_edge !=0:
  
        agg_weight = debt_edge + cds_edge
  
        weighted_aggregated_network[i][j] = agg_weight   
  
  
  result = [debt_network, cds_network, weighted_aggregated_network]
  
  avg_cds_debt = np.average(cds_over_debts)
  
  max_cds_debt = np.max(cds_over_debts)
  
  std_cds_debt = np.std(cds_over_debts)

  return result, num_naked_cds,tot_cont, num_elem_cycles, avg_cds_debt, max_cds_debt,std_cds_debt




def network_meassures(net):

  """Return the comma separate string of calculated metrics


  type : string

  Parameters
  ----------
  net : 2d array  ( adjacency matrix (weighted or non-weighted))

  """

  g, edge_weights = convert_to_graphtool_object(net)
  g_nx = convert_to_networkx_object(net)



  #degree related metrics

  w_total_degrees=(graph_tool.Graph.degree_property_map(g, deg="total", weight=edge_weights))
  total_degrees=(graph_tool.Graph.degree_property_map(g, deg="total"))
  
  w_d_avg, w_d_std = graph_tool.stats.vertex_average(g, w_total_degrees)
  d_avg, d_std = graph_tool.stats.vertex_average(g, total_degrees)
  

  in_assortivity = np.average(graph_tool.correlations.scalar_assortativity(g, 'in'))
  out_assortivity =np.average(graph_tool.correlations.scalar_assortativity(g, 'out'))
  total_assortivity = np.average(graph_tool.correlations.scalar_assortativity(g, 'total'))
  
  assortivity = nx.degree_assortativity_coefficient(g_nx)

  vertices = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())

  sizes, comp = graph_tool.topology.vertex_percolation(g, vertices)

  np.random.shuffle(vertices)
  sizes2, comp = graph_tool.topology.vertex_percolation(g, vertices)
  
  perlocution_size = np.average(sizes)

  perlocution_size2 = np.average(sizes2)


  
  #katz centrality measures

  katz_centrality=(graph_tool.centrality.katz(g))
  w_katz_centrality = graph_tool.centrality.katz(g, alpha=0.00001, weight=edge_weights, max_iter=100000)
  average_katz, std_katz =graph_tool.stats.vertex_average(g, katz_centrality)
  average_w_katz, std_w_katz =graph_tool.stats.vertex_average(g, w_katz_centrality)


  
  #eigenvalue centrality measures

  max_w_eigenvalue, w_eigenvector = graph_tool.centrality.eigenvector(g, max_iter=100000)

  average_w_eigenvalue, std_w_eigenvalue = graph_tool.stats.vertex_average(g, w_eigenvector)
# Hub and Authority scores

  w_e, w_authority, w_hub = graph_tool.centrality.hits(g, weight=edge_weights, max_iter=100000)
  average_w_hub, std_w_hub = graph_tool.stats.vertex_average(g, w_hub)

  
  return str(w_d_avg)+","+str(w_d_std)+","+str(d_avg)+","+str(d_std)+","+str(in_assortivity)+","+str(out_assortivity)+","+str(total_assortivity)+","+str(assortivity)+","+str(perlocution_size)+","+str(perlocution_size2)+","+str(average_katz)+","+str(std_katz)+","+str(average_w_katz)+","+str(std_w_katz)+","+str(average_w_eigenvalue)+","+str(std_w_eigenvalue)+","+str(average_w_hub)+","+str(std_w_hub)+","
  

def membership_coefficient(aggregated_network, debt_network, cds_network):
  """Return the graph object associated with the adjacency matrix of a network,as well
    as its corresponding weights


  type : vector of membership coefficient for each entity 

  Parameters
  ----------
  aggregated_network : 2d array  ( adjacency matrix (weighted or non-weighted))
  debt_network: 2d array
  cds_network:  2d array
  """

  p = [ 0 for i in range(len(debt_network))]
  
  g_debt, edge_weights_debt = convert_to_graphtool_object(debt_network)
  g_cds, edge_weights_cds = convert_to_graphtool_object(cds_network)
  g_agg, edge_weights_agg = convert_to_graphtool_object(aggregated_network)

  total_degrees_debt= graph_tool.Graph.degree_property_map(g_debt, deg="total")
  total_degrees_cds= graph_tool.Graph.degree_property_map(g_cds, deg="total")
  total_degrees_agg= graph_tool.Graph.degree_property_map(g_agg, deg="total")
  
  for i in range(len(debt_network)):
  
      SUM = 0	
  	  
      SUM += (total_degrees_debt[i]/total_degrees_agg[i]+0.0)**2
      SUM += (total_degrees_cds[i]/total_degrees_agg[i]+0.0)**2
      p[i] = (1-SUM)

  return p


# leverage of network
def frag_index(networks, e):
  """Return leverage measure of different subnetworks as well as overall network

  type : array of lenght 3, each element corresponds to the leverage measure of a subnetwork   

  Parameters
  ----------
  networks : array of 2d arrays
  e : array (external assets)
  
"""

  h =[]
  n_tot = 0
  d_tot = 0
  n = len(networks[0]) # num of nodes, banks


  for network in networks:
    for i in range(n):
      if e[i]!=0:
    	  debt_i = sum(network[i][:])
    	  n_tot += debt_i
    	  d_tot += debt_i**2/e[i]

    h.append((n_tot**2/d_tot))

    n_tot = 0
    d_tot = 0	

  return h 	


def  Herfindhal_index(networks):

  """Return the inverse of Herfindhal index of different subnetworks as well as overall network

  type : 2d array   

  Parameters
  ----------
  networks : array of 2d arrays
  e : array (external assets)
  
"""


  h =[]
  n_tot = 0
  d_tot = 0
  n = len(networks[0]) # num of nodes, banks


  for network in networks:
    for i in range(n):
  	  debt_i = sum(network[i][:])
  	  n_tot += debt_i+0.0
  	  d_tot += debt_i**2+0.00


    h.append(n_tot**2/d_tot)

    n_tot = 0
    d_tot = 0	

  return h  

def separate_effect_measures(networks):

  """Apply 'network_meassures(net)' on each subnetwork, store the resutl in a comma separated string 
  
  type : string

  Parameters
  ----------
  networks : array of 2d arrays
  
"""



  measures_so_far = ""

  pc = membership_coefficient(networks[2], networks[0], networks[1])
  
  avg_pc, std_pc = np.average(pc), np.std(pc)

  hii = Herfindhal_index(networks)  
  hii_debt, hii_cds, hii_agg = hii[0], hii[1], hii[2]
  
  for net in networks:
    
    measures_so_far += network_meassures(net)
      
    
  return measures_so_far + str(avg_pc)+","+str(std_pc)+","+str(hii_agg)+","+str(hii_debt)+","+str(hii_cds) 	
  