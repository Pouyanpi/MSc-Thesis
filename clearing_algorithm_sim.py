import itertools
import operator
import pandas as pd
import numpy as np
from amplpy import AMPL, DataFrame
import update_func as uf
import net_to_polynomials






# # 
def array_to_ampl_dataframe(n_banks, liability): #change the name of liability to 2d array? TODO
    ampl_liab = DataFrame(('Banks','Banks2'),'liability') #TODO : set 'liability' to parameter name, can get it? 
    
    
    ampl_liab.setValues({
        (i, j): liability[i][j]
        for i in xrange(n_banks)
        for j in xrange(n_banks)
    })
    
    return ampl_liab


def clearing_algorithm(id,v_e, li, c, ref_entities):


    """Stress testing algorithm : Solve the clearing problem.


    type : number of infeasible solutions
         num of non maximal solutions
          recovery rate vector
         equity
          vector of error in the update function  

    Parameters
    ----------
    id : string 
    v_ea : 2d array  ( post externall assets; vector of assets after shock)
    li : 2d array (liability matrix, debt contract)   
    c : 3d array (cds contracts)
    ref_entities : list (set of reference entities)


    """

    id_file_nonmax = open('id_file_nonmax.txt', 'a')
    id_file_infeasible = open('id_file_infeasible.txt','a')

    nBanks = len(v_e[0])
    fixed_vec = list()
    alpha, beta = 0.6, 0.6
    externalAssets,liability,CDS_contract,reference =v_e[0],li,c,ref_entities

    ## call AMPL 

    ampl = AMPL()

    # set default cost parameters

    ampl.read('models/clearing_optimization.mod') #c2.mod
    ampl.readData('data/clearing_optimization.dat')# change this to irrational.dat if you don't have default costs
    
    #Set ampl options

    ampl.setOption('display_precision',0)
    ampl.setOption('solution_precision',0)

    # we will use Knitro as the solver, other options: lgo, loqo, conopt, ...

    ampl.setOption('solver', 'knitro')
    ampl.setOption('presolve_eps',1.0e-6)
    ampl.setOption('outlev',0)
    ampl.setOption('show_stats',0)

    # set knitro options

    ampl.setOption('knitro_options', 'par_msnumthreads =32 ms_maxsolves=1 outlev=0 ma_terminate = 2 feastol = 1.0e-9 infeastol= 1.0e-1 feastol_abs= 1.0e-9 ms_deterministic=0  ma_terminate = 1 ma_outsub =1 bar_refinement=1  bar_slackboundpush=1.0 delta=1.0e-1 gradopt=1 hessopt=1 honorbnds=0 linesearch=0 linsolver_ooc=0 ms_enable=1 tuner=0 soc=0 initpenalty=3.0e10 bar_penaltyrule=1')# derivcheck=3')

    # another set of options to use, the above options has been tested and compared to other optio sets,
    # it obtained the most accurate results
    # one can use his/her options : see 

    # ampl.setOption('knitro_options', 'par_msnumthreads =16 ms_maxsolves=10  ma_terminate = 2 feastol = 1.0e-9 infeastol= 1.0e-1 feastol_abs= 1.0e-9 ms_deterministic=0  ma_terminate = 1 ma_outsub =1 bar_refinement=1  bar_slackboundpush=1.0 delta=1.0e-1 gradopt=1 hessopt=1 honorbnds=0 linesearch=0 linsolver_ooc=0 ms_enable=1 tuner=0 soc=0 initpenalty=3.0e10 bar_penaltyrule=1 derivcheck=3')#    out_csvinfo=1   restarts=10 ms_maxsolves=0  soc=1 tuner=1 #bar_relaxcons=2 #bar_watchdog=1
    
    solver_status_maximal =""


### prepare ampl, and initialize it by setting the data


    ampl_alpha = ampl.getParameter('alpha')
    ampl_beta = ampl.getParameter('betta')

    ampl_alpha.setValues(alpha)
    ampl_beta.setValues(beta)

    banks = [i for i in xrange(nBanks)] #vector of indices

    ampl.getSet('Banks').setValues(banks)
    ampl.getSet('reference').setValues(d)


    ampl_liab = array_to_ampl_dataframe(nBanks,liability)

    ampl.setData(ampl_liab)

 
    ampl_external_assets = DataFrame('Banks', 'externalAssets')
    ampl_external_assets.setColumn('Banks', banks)
    ampl_external_assets.setColumn('externalAssets', externalAssets)
    
    ampl.setData(ampl_external_assets)
    
    
    ampl_CDSs = DataFrame(('i','j','k'), 'CDS_contract')
    for i in xrange(nBanks):
    	for j in xrange(nBanks):
    		for k in d:
    			ampl_CDSs.addRow(i,j,k,c[i][j][k])
        

    ampl.setData(ampl_CDSs)

    maximal_out = []
    infeasible_out = []
    total_recovery_rates = []
    total_equity = []
    f_vec = []  

    """
    set the objective, named recov
    if we have this objective funtion
    then the problem become Clearing Feasibility Problem
    as the value of recovery_rate_no_debts is constant
"""
    ampl.eval('maximize recov : sum{i in Banks} recovery_rate_no_debts[i];')
    
    ''' 
        for each element of the post external asset we need to solve the clearing problem
        remark: post_externall_assets are defined in the cl_btie_main.py or cl_uni_main.py;
        they contain external assets after applying shocks
        since we need to run the clearing algorithm for various array of external assets (see shock_gen.py) 
        and we don't want to have AMPL's loading and initializing overhead at every time.
        we will just update the externall assets parameter at each round.'''
    
    for m in range(len(v_e)):

        # if external asset is zero then, obviously, we don't do the clearing:
        # shock on a bank without externall assets does not change the vector of externall assets 
        
        if v_e[0][m] !=0:


            equity = [1000000000000 for i in range(nBanks)]
            
            # set value of previous equity to infinity as we want this constraint be redundant in solving
            # the Clearing Optimization Problem and checking if the problem is infeasible
            
            prev_eq = ampl.getParameter('preveq') #.getValues()

            prev_eq.setValues(equity)
            # drop the Clearing Optimization objective
            ampl.obj['recov'].drop()
            # restore new objective which is constant: to use if for the Clearing Feasibility Problem, and checking maximality
            #Solve the clearing optimization problem,
            # we solve the model given our data, but this time the objective function is maximizing 'total_equity'
            # as defined in the clearing_optimization.mod .

            ampl.obj['Tot_recov'].restore()
            
            ea = ampl.getParameter('externalAssets')

            ea.setValues(v_e[m])  

            # set ampl options again
            
            ## Clearing Optimization Problem, checking feasibility

            ampl.setOption('knitro_options', 'par_msnumthreads =32 ms_maxsolves=10 outlev=0 ma_terminate = 2 feastol = 1.0e-9 infeastol= 1.0e-1 feastol_abs= 1.0e-9 ms_deterministic=0  ma_terminate = 1 ma_outsub =1 bar_refinement=1  bar_slackboundpush=1.0 delta=1.0e-1 gradopt=1 hessopt=1 honorbnds=0 linesearch=0 linsolver_ooc=0 ms_enable=1 tuner=0 soc=0 initpenalty=3.0e10 bar_penaltyrule=1')# derivcheck=3')
          
            ampl.solve()

            tot_payment_1 = ampl.obj['Tot_recov']

            solver_status = tot_payment_1.result()

            tot_payment_1 = tot_payment_1.drop()
            
            ampl.obj['Tot_recov'].drop()
            
            ampl.obj['recov'].restore()

          
            recoveryRate = ampl.getVariable('result').getValues().toList()
            equity = (ampl.getVariable('equity').getValues()).toList()
            
            ## update recovery results by rounding those near to 1, to 1

            for x in xrange(len(recoveryRate)):

                if recoveryRate[x][1]>1 or recoveryRate[x][1] > 0.99999999:
                    recoveryRate[x] = 1
                else :
                    recoveryRate[x] = (recoveryRate[x])[1]    

            for x in range(len(equity)):
                equity[x] = equity[x][1]

            '''
            #retrieve alpha and beta
            
            alpha = ampl.getParameter('alpha').getValues()
            alpha = ((alpha.toList())[0][0])
            beta = ampl.getParameter('betta').getValues()
            beta = ((beta.toList())[0][0])
      

            '''    

            
            CDSs = d
   
            # s is the result of update function, i.e., the difference of approximate and actual value
            s =abs(uf.update_f(alpha,beta,nBanks, CDSs,v_e[m],b,c, recoveryRate))
   
            if solver_status =='solved':

            ## CLearing Feasibility Problem (maximality check)    
                maximality = 'maximal'
                prev_eq.setValues(equity)
            
                ampl.setOption('knitro_options', 'par_msnumthreads =32 ms_maxsolves=1 outlev=0 ma_terminate = 2 feastol = 1.0e-9 infeastol= 1.0e-1 feastol_abs= 1.0e-9 ms_deterministic=0  ma_terminate = 1 ma_outsub =1 bar_refinement=1  bar_slackboundpush=1.0 delta=1.0e-1 gradopt=1 hessopt=1 honorbnds=0 linesearch=0 linsolver_ooc=0 ms_enable=1 tuner=0 soc=0 initpenalty=3.0e10 bar_penaltyrule=1')# derivcheck=3')
            

            ## solve the clearing feasibility problem, this time to check if the previous solution is maximal
            # if it returns infeasible, then the solution of Clearing Optimization program is not maximal, other wise
            # we have found a maximal solution
        

                ampl.solve()
                tot_payment_2 = ampl.getObjective('recov')

                solver_status_maximal = tot_payment_2.result()

                tot_payment_2 =tot_payment_2.drop()

                if solver_status_maximal=='solved':
                    maximality = 'non_maximal'
                else :
                    maximality = 'maximal'    
          
            else:
                maximality ='none'  


            total_equity.append(sum(equity))                                
            total_recovery_rates.append(np.count_nonzero(np.array(recoveryRate)-1))
            
            if solver_status !='infeasible':
                f_vec.append(s)    

            if maximality =='non_maximal':

                maximal_out.append(m)
                status = 'nonmax' 
                id_file_nonmax.write(id)
                generate_polynomials(status,id,v_e[m], li, c, ref_entities, alpha, beta)

            if solver_status =='infeasible' : 
                
                infeasible_out.append(m)
                status= 'infeasible'
                id_file_infeasible.write(id)
                generate_polynomials(status,id,v_e[m], li, c, ref_entities, alpha, beta)

             



    ampl.reset()

    return f_vec, total_recovery_rates, total_equity, infeasible_out, maximal_out
