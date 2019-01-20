import sys
import bowtie_network_generator as bg
import network_measures as ml
import networkx as nx
import clearing_algorithm_sim as cl
import math
import random
import numpy as np
import os
import errno
from extergen import *
import shock_gen as shck

n_banks =50
p = 0.25

alpha, beta = 0.6, 0.6

levRatio =[0.02, 0.052, 0.12] 

lossRate =[0.4, 0.6, 0.95]

num = int(sys.argv[4])

### getting the rest of parameters from console. To be used on cluster

netbuyerfraction, netsellerfraction, pdd = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

'''
maximum and minimum number of iterations,

here 5 times the code is executed within each iteration

'''
min_iter, max_iter = 5*num, 5*(num+1) 

# vectors keeping non maximal and infeasible solutions returned by clearing algorithm,
# f_vec is the vector of averaged errors in the update function, i.e., the difference between actual and the approximate value (see update_func.py)

maximal_out, infeasible_out, f_vec = [],[],[] 



# do the simulations

while min_iter < max_iter:

    # generate finacnial networks with CDSs based on bowtie_network_generator.py

    li, c, cds_banks= bg.bowtie_network_gen(n_banks, netbuyerfraction, netsellerfraction, 0.1, pdd)
    
    # for understanding these metrics se bowtie_network_generator.py

    netbuyers = math.floor(n_banks * netbuyerfraction)
    netsellers=math.floor(n_banks * netsellerfraction)
    total_netbuyers_netsellers=netbuyers+netsellers
    dealers=n_banks-total_netbuyers_netsellers

    # density of the SCC, core component : based on the statistics defined in the report
    dens = 0.1

    j = min_iter

    min_iter+=1

    w, nei, size = 1, math.ceil(dealers*pdd), dealers
    id_result = "result_bowtie_it_"+str(j)+"_"+str(w)+"_"+str(netbuyerfraction)+"_"+str(netsellerfraction)+"_"+str(dens)+"_"+str(pdd)+".txt"
    filename ="outcomebowtie"+str(w+netbuyerfraction+netsellerfraction+pdd)+"_nei_"+str(w)+"_beta_"+str(beta)+"/"+id_result

    #calculate network measures

    networks, n_naked, n_cont, n_cycles, avg_cds_debts, max_cds_debts, std_cds_debts = ml.convert_to_sep_networks(li, c, cds_banks)

    measures_so_far = ml.separate_effect_measures(networks)+","+str(n_naked)+","+str(n_cont)+","+str(n_cycles)+","+str(avg_cds_debts)+","+str(max_cds_debts)+","+str(std_cds_debts)

    
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise    


    FILE = open(filename, 'a')

    for leverageratio in levRatio:
    
        # generate external asset give leverage ratio 

        ea, debts_over_assets = external_gen(li, c, leverageratio)

        # get frag index: leverage of network  for different subnetworks

        frag_indices = ml.frag_index(networks, ea)
        
        fi_debt, fi_cds, fi_agg = frag_indices[0], frag_indices[1], frag_indices[2] 
        
        avg_fi = np.mean(frag_indices)

        # dos : debts over assets, it is specified in the extergen.py 

        avg_dos, std_dos, max_dos = np.average(debts_over_assets), np.std(debts_over_assets), np.max(debts_over_assets)
        
        sum_ea, avg_ea, st_ea = sum(ea), np.average(ea), np.std(ea)
        
        
        for rate in lossRate:


            total_equity, total_recovery_rates = [], []

            post_external_assets = np.asarray(shck.shock_gen(n_banks, ea, rate)).tolist()
            
            id1 = filename+"_"+j+"_lr_"+str(rate)+"_lvr_"+str(leverageratio)
            
            id_output = id1+","+str(p)+","+str(leverageratio)+","+str(rate)
           
            ## call the stress testing algorithm , clearing_algorithm(), and retrieve related results

            f_vec, total_recovery_rates, total_equity, infeasible_out, maximal_out =cl.clearing_algorithm(id_output,post_external_assets, li, c, cds_banks)

            to_csv = id_output+","+measures_so_far+","+str(avg_dos)+","+str(std_dos)+","+str(max_dos)+","+str(sum_ea)+","+str(avg_ea)+","+str(std_ea)+","+str(fi_debt)+","+str(fi_cds)+","+str(fi_agg)+","+str(avg_fi)+","+str(np.average(total_equity))+","+str(np.std(total_equity))+","+str(np.average(total_recovery_rates))+","+str(np.std(total_recovery_rates))+","+str(np.average(f_vec))+","+str(len(maximal_out))+","+str(len(infeasible_out))+"\n"
            
            # write results to file, outcome is a csv file        
            
            FILE.write(to_csv)    

