
# Getting Started

This file contains an overview of the components of the **Analysis Framework** (see Chapter 3 from Pouyan Rezakhani, 2019). Detailed description of the codes and also installation Guides are provided. This guide assumes that the user has downloaded the project source codes and has already read the related report.

## Installation


**Install Python3.7, MATLAB, and AMPL**:


In Linux :
```bash
    sudo apt-get update
    sudo apt-get install python3.6
```
In Windows :    

   * Open a browser window and navigate to the Download page for Windows at python.org.
   * Underneath the heading at the top that says Python Releases for Windows, click on the link for the Latest Python 3 Release - Python 3.6 .
   
   * Scroll to the bottom and select either Windows x86-64 executable installer for 64-bit or Windows x86 executable installer for 32-bit. (See below.)


**Install AMPL** :


In Linux:

To isntall latest version of AMPL (A Mathematical Programming Language):

Just download it from the following [link](https://ampl.com/products/ampl/ampl-for-students/) ( if you don't have the full academic license, then one might consider using student license)


**Install MATLAB**

1. Download [Software](https://www.mathworks.com/login?uri=%2Fdownloads%2Fweb_downloads%2Fselect_release%3Fmode%3Dgwylf%26refer%3Dmwa%26s_cid%3Dmwa-cmlndedl)
2. Choose the version for your computer
3. Start the installer
4. Activate your installed version of Matlab ( for further infromation regarding license and activation key for UZH students and staffs see this [link](https://www.id.uzh.ch/en/dl/sw/angebote/statmath/matlab.html)
5. Agree to the terms and conditions !

**Install Anaconda**

It is recommended to install Anaconda as it will make the process of other installations in this work easires. 

Download [conda](https://www.anaconda.com/download/) installer.
``` bash        
    bash Anaconda-latest-Linux-x86_64.sh
```

For further information about Installing packages with conda follow the [link](https://conda.io/docs/commands/conda-install.html).    

    
The rest of the installation guide would be based on the contents and different modules of the project. 


# Content


## Stress Testing Algorithm


### Clearing Algorithms

To implement the Clearing Optimization (Feasibility) Problem, we have used AMPL modeling language. In order to make use of the clearing algorithms one need to install the AMPL python api (amplpy). In this section we will firstly provide instructions on the installation and then explain components of clearing algorithm.

To install **amplpy** with pip :

```bash
   
   python -m pip install amplpy

```
<div class="alert alert-success">
It is important to note which python version is <code>pip</code> command linked to, here it is assumed it is linked to python3; however, if one is linked to python 2, one can install <code>pip3</code>. 
</div>

Having installed the amplpy, we can examine the clearing algorithm which is provided in the *clearing_algorithm_sim.py*.

To make this code work following modules should get imported:



```python
import itertools
import operator
import pandas as pd
import numpy as np
from amplpy import AMPL, DataFrame
import update_func as uf
```

It is supposed that numpy and pandas are both already installed. However, one can isntall them using pip :

If some one needs to install **numpy** :

With **anaconda**:
* If you have already installed anaconda, then it is included.

With **pip** (again "python" command depends on the python version being used by the user):

```bash
    python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

We have already install pandas with use of above commands; however, to install pandas singly:

```bash
   pip install pandas
   
```

Now we will briefly explain the last module imported, i.e., update_func:

This code, update_func.py, will calculate the difference from a given solution and the actual result in the Update function, that is $F(r)-r = \epsilon$ . It is used as a measure to check the accuracy of a result.


<div class="alert alert-block alert-info">
<b><code>update_f(alpha,beta,n_banks,reference_entities,ea,debt_contract,cds_contract, recovery_rates)</b></code>    
    
    Return epsilon as defined above.
<b>type:</b>
    
    float
    
<b>Parameters:</b> 
    
    alpha, beta : int, Default cost parameters
    n_banks : int , number of banks
    reference_entities: list, set of reference entities    
    ea : array, external assets
    debt_contract : 2d-array
    cds_contract: 3d-array>
    recovery_rates : array
    
</div>


For better illustration, one can see the detail of this method as :

``` python

def update_f(alpha,beta,n_banks,reference_entities,ea,debt_contract,cds_contract, recovery_rates):
	
	fixed_point_difference = [0 for i in range(n_banks)]
	li = liabilities(n_banks,reference_entities, debt_contract,cds_contract,recovery_rates)

	total_li = total_liability(n_banks,li)
	payments = payment(n_banks,li, recovery_rates)
	total_assets = total_assets(n_banks,payments,ea,total_li,alpha,beta)

	for i in xrange(n_banks):
		if total_assets[i]< total_li[i] :

			fixed_point_difference[i]=((total_assets[i]+negligible)/total_li[i])-recovery_rates[i]

		else:

			fixed_point_difference[i] = (1- recovery_rates[i])

	epsilon = sum(fixed_point_difference)/len(fixed_point_difference) 

	return epsilon

```
Now we will take a closer look into the *clearing_algorithm_sim.py* :



<div class="alert alert-block alert-info">
    <b><code>clearing_algorithm(v_e, li, c, ref_entities)
        </b></code>
    
    
    Stress testing algorithm : Solve the clearing problem.


<b>type</b> : 

       <type 'int'> (number of infeasible solutions)
       <type 'int'> (num of non maximal solutions)
       <type 'list'> (recovery rate vector)
       <type 'list'> (equity)
       <type 'list'> (vector of error in the update function)

<b>Parameters:</b>
        
        id : string (id of the data generated, used for saving results in 'net_to_polynomials.py')
        v_ea : 2d-array, ( post externall assets; vector of assets after shock)
        li : 2d-array, (liability matrix, debt contract)   
        c : 3d-array, (cds contracts)
        ref_entities : list, (set of reference entities)


    
</div>


First we need to call AMPL, then we will choose the solver we want to use, here Knitro, and we will set desire options :







```python
## call AMPL 
ampl = AMPL()

# set default cost parameters

ampl.read('models/clearing_optimization.mod') # clearing model
ampl.readData('data/clearing_optimization.dat') # sample data to build our model on top of it
#Set ampl options

ampl.setOption('display_precision',0)
ampl.setOption('solution_precision',0)
# we will use Knitro as the solver, other options: lgo, loqo, conopt, ...

ampl.setOption('solver', 'knitro')
ampl.setOption('presolve_eps',1.0e-6)
ampl.setOption('outlev',0)
ampl.setOption('show_stats',0)
# set knitro options

ampl.setOption('knitro_options', 'par_msnumthreads =32 ms_maxsolves=1 outlev=0 ma_terminate = 2 \
feastol = 1.0e-9 infeastol= 1.0e-1 feastol_abs= 1.0e-9 ms_deterministic=0  ma_terminate = 1\
ma_outsub =1 bar_refinement=1  bar_slackboundpush=1.0 delta=1.0e-1 gradopt=1 hessopt=1\
honorbnds=0 linesearch=0 linsolver_ooc=0 ms_enable=1 tuner=0 soc=0\
initpenalty=3.0e10 bar_penaltyrule=1')

```

We have set the above options after testing a large number of samples. This solver was chosen among many other solvers such as : conopt, snopt, loqo, lgo, minos, etc. The parameters are defined carefully with regard to the clearing_optimization.mod, they have been tested as well. For details Knitro options see its [page] (https://www.artelys.com/tools/knitro_doc/3_referenceManual/knitroamplReference.html#knitroamplreference).

The parameter that is recommended to vary given your computation resources is "ms_maxsolves", which declares maximum number of start points to try during multi start, basically, the solver with run the program upto this number within each iteration. Higher this value will increase the accuracy of the results, however, it should be chosen cautiosly. (recommended : not above 100)

After setting solver options we will initialize model's parameters ( see *clearing_optimization.mod*) with respect to financial systems' data. (see the *clearing_algorithm_sim.py* for detailed description)

The algorithm follows the following steps:

1. since we need to run the clearing algorithm for various array of external assets (see shock_gen.py) and we don't want to have AMPL's loading and initializing overhead at every time. we will just update the externall assets parameter at each round.



2. Solve the clearing optimization problem, we solve the model given our data, but this time the objective function is maximizing 'total_equity' as defined in the clearing_optimization.mod .

    - we will set the 'prev_eq' parameter to a large number, to make the 'maximality' constraint in the model redundant. This constraint is used for checking the maximality condition

3. we store the solver status at this round ('solver_status')
    - it can be 'solved'
    - it can be 'infeasible'
    - if 'solver_status' is 'infeasible' then we will call the '.py' which we will explain it later

4. we will set 'prev_eq' to the 'equity' obtained from calling AMPL.solve() at the previous round. Thus, the 'maximality' constraint is not redundant anymore.

5. we will drop the previous objective function and set the new objective to a constant ( we solve the feasibility problem this time)



6. we store the solver status at this round ('solver_status')
    - it can be 'solved'
    - it can be 'infeasible'
    - if 'solver_status' is 'infeasible' then we will call the '.py' which we will explain it later

We will present the model used for this problem (clearing_optimization.mod):

```python


#clearing algorithm for financial networks with CDSs 
# this modeling is used for MSc thesis of Pouyan Rezakhani

set Banks;

set reference within Banks ;

param externalAssets {i in Banks} >= 0 ;

param liability {i in Banks, j in Banks} >=0;

param CDS_contract  {i in Banks, j in Banks, k in reference} >= 0;

set Writers = {i in Banks: exists {j in Banks, k in reference}   (liability[i,j]>0 or CDS_contract[i,j,k]>0)};
set Holders = {j in Banks: exists {i in Banks, k in reference}  (liability[i,j]>0 or CDS_contract[i,j,k]>0)};
set noDebt = {i in Banks: forall {j in Banks, k in reference} liability[i,j]=0 and CDS_contract[i,j,k]=0};
set only = {i in Banks: forall {j in Banks, k in reference} liability[i,j]=0 };
set noCDS = {i in only :exists {j in Banks, k in reference} CDS_contract[i,j,k]>0};
set intesct = Writers union Holders diff noDebt;

# recovery rate of banks with no obligations

param recovery_rate_no_debts {i in Banks} =  if i in noDebt then 1 else 0  ;

# set preveq 
param preveq {i in Banks} default -1000000;
param tot_preveq = sum{i in Banks} preveq[i];
#Default cost parameters 
param alpha  in [0,1]; 
param betta in [0,1];

#inputs to the clearing problem :


var recovery_rate { i in Banks} in [0,1];#:= 1;

var CDS_Liability {i in Writers, j in Holders, k in reference : CDS_contract[i,j,k] >0 and i<>j and j<>k and i<>k} =
 (1- (recovery_rate[k]+recovery_rate_no_debts[k]) ) * CDS_contract[i,j,k];

var payment {(i,j) in Banks cross Banks : i<>j} =(recovery_rate[i]+  recovery_rate_no_debts[i])* ( liability[i,j] + ( sum{k in reference :CDS_contract[i,j,k] >0 and i<>j and j<>k and i<>k} CDS_Liability[i,j,k]));

var Liabilities {i in Banks} = sum{j in Banks : i<>j} liability[i,j] + sum{ j in Holders, k in reference : CDS_contract[i,j,k] >0 and i<>j and j<>k and i<>k} CDS_Liability[i,j,k] ;#+ 0.0000000000000000000000000000000000000000000000000000000000001;

var inPayments {i in Banks} = sum {j in Writers: i<>j} payment[j,i];

var outPayments {i in Banks} = sum {j in Holders: i<>j} payment[i,j];

var equity {i in Banks} = externalAssets[i] + inPayments[i] -Liabilities[i] ;

var total_equity = sum{i in Writers} equity[i];

var isDefault {i in Writers } = if equity[i] >=-0.0000000001 then 0 else 1 ; 

subject to bankruptcy_rules {i in Writers } :
	recovery_rate[i]=min ( ( ( ( (isDefault[i]))*(alpha-1) + 1)*externalAssets[i] + ( ( ( (isDefault[i]))*(betta-1) + 1)*inPayments[i] ) ) /Liabilities[i], 1 );
	

subject to maximality{ i in Writers}:

	equity[i] <=  preveq[i] - 0.0000000000001; 

	
var result {i in Banks} = recovery_rate[i] + recovery_rate_no_debts[i];

maximize Tot_eq : total_equity;


```

Simply we have implemented the "clearing optimization problem" (see Section 2.3.1)


Furthermore, we have also used another code *net_to_polynomials.py*. This file will take all information of a financial network and provide the symbolic expressions and polynomials required for generating certificates. (see Section 2.3.3). It will store the results in matlab files with '.mat' extension. Later, we will use it within our MATLAB code.

<div class="alert alert-block alert-info">

<b><code>generate_polynomials(status,id,ea, li, c, cds_banks, alpha, beta)</code></b>

    Save polynomials in MATLAB symbolic expressions into MATLAB files.
    

<b>Parameters:</b>
    
    status : string (decide under which file to save the results)
    id : string (id of the infeasible data) 
    ea : array (vector of external assets)
    li : 2d-array ( matrix of debt contracts)
    c : 3d-array ( cds contracts)
    cds_banks : list (reference entities)
    alpha : float (default parameter)
    beta : float (default parameter)
</div>

Please refer to the code for its detail.


## Shock Generator

Shock generator is discussed in section 3.2.4 of the report. 


<div class="alert alert-block alert-info">

<b><code>shock_gen(nbanks, ea, lr)</code></b>

    Return the 2d-array of external assets after applying shock with given loss rate (lr).
    
<b>type</b> : 
    
    2d-array

<b>Parameters:</b>

    nbanks : int (number of banks)
    ea : array (vector of external assets)
    lr : float ( loss rate)
</div>

For illustrative purposes it is provided here.



```python
from shock_gen import *

lr = 0.35 # loss rate 
ea = [12, 32, 4, 5]
nbanks = len(ea)

print shock_gen(nbanks, ea , lr)
```

    [[ 7.8  32.    4.    5.  ]
     [12.   20.8   4.    5.  ]
     [12.   32.    2.6   5.  ]
     [12.   32.    4.    3.25]]
    


## Network Measures

As described in section 4.1 we will define various measures here with respect to different subnetworks. To this end, we will define different subnetworks as well as the overall network, colored dependency graph representation. Afterwards, we will apply the provided metrics in the '*network_measures.py*' on each of them. We have also provided some other metrics.

This module is provided in network_measures.py . To make this module works it is necassary to install graph_tool .[Graph-tool](https://graph-tool.skewed.de/) is an efficient Python module for manipulation and statistical analysis of graphs. In contrast to most other python modules with similar functionality, the core data structures and algorithms are implemented in C++, making extensive use of template metaprogramming, based heavily on the Boost Graph Library. This confers it a level of performance that is comparable (both in memory usage and computation time) to that of a pure C/C++ library.
For Linux (Ubuntu), add the following lines to the source file (i.e. source.list).
``` bash
    deb http://downloads.skewed.de/apt/DISTRIBUTION DISTRIBUTION universe
    deb-src http://downloads.skewed.de/apt/DISTRIBUTION DISTRIBUTION universe
```    
where **DISTRIBUTION** can be any one of

    xenial, yakkety, zesty, artful, bionic, cosmic

After running apt-get update, the package can be installed with

```bash    
    sudo apt-get update
    sudo apt-get install python-graph-tool
```
or if you want to use Python 3
``` bash    
    sudo apt-get install python3-graph-tool
```
Detailed installation is provided in this [link](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions)


With **Conda** : 

```bash
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
    
```
In addition to graph-tool, we also used 'networkx' package. We have used this package as networkx outperforms graph-tool in some measures and some metrics are not implemented in graph-tool.

To install Networkx:

```bash
pip install networkx

or

ip install --user networkx # if you don't have priviledged access on the cluster

```

Following import process is necessary :


```python
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
```

The description of the methods used in this module are provided below.

<div class="alert alert-block alert-info">

<b><code>convert_to_networkx_object(net)</code></b>

    Return the graph object associated with the adjacency matrix of a network
    
<b>type</b> : 
    
    networkx Graph

<b>Parameters:</b>

    net : 2d array  ( adjacency matrix (weighted or non-weighted))


    
***
<b><code>convert_to_networkx_object(net)</code></b>


<b>type</b> : 
  

<b>Parameters:</b>


***
<b><code>convert_to_graphtool_object(net)</code></b>

    Return the graph object associated with the adjacency matrix of a network,as well as its corresponding weights

<b>type</b> : 

    graph-tool Graph
         2d array 

<b>Parameters:</b>

    net : 2d array  ( adjacency matrix (weighted or non-weighted))

***

<b><code>convert_to_sep_networks(li, c, ref_entities)</code></b>

       Return different subnetworks, i.e., CDS and debt subnetwork; and the overall network
       +
       number of naked CDSs  (int)
       number of all contracts (int)
       number of red only cycles (int)
       average, std, and maximum value of cds to debt ratio (float)

<b>Parameters:</b>

      li : 2d array (liability matrix, debt contract)   
      c : 3d array (cds contracts)
      ref_entities : list (set of reference entities)
***
<b><code>network_meassures(net)</code></b>
    
    Return the comma separate string of calculated metrics
    
    list of metrics : w_d_avg, w_d_std, d_avg, d_std, in_assortivity, out_assortivity, total_assortivity, assortivity, perlocution_size, perlocution_size2, average_katz, std_katz, average_w_katz, std_w_katz, average_w_eigenvalue, std_w_eigenvalue, average_w_hub, std_w_hub


<b>type</b> : 
      
      string

<b>Parameters:</b>
    
    net : 2d array  ( adjacency matrix (weighted or non-weighted))

***

<b><code>membership_coefficient(aggregated_network, debt_network, cds_network)</code></b>

    Return the graph object associated with the adjacency matrix of a network,as well
    as its corresponding weights



<b>type</b> : 
     
     vector of membership coefficient for each entity 

<b>Parameters:</b>

    aggregated_network : 2d array  ( adjacency matrix (weighted or non-weighted))
    debt_network: 2d array
    cds_network:  2d array

***



<b><code>frag_index(networks, e)</code></b>

    Return leverage measure of different subnetworks as well as overall network


<b>type</b> : 

    array of lenght 3, each element corresponds to the leverage measure of a subnetwork   

  

<b>Parameters:</b>

    networks : array of 2d arrays
    e : array (external assets)
***






<b><code>Herfindhal_index(networks)</code></b>
    
    Return the inverse of Herfindhal index of different subnetworks as well as overall network


<b>type</b> : 
    
    2d array 
  

<b>Parameters:</b>

    networks : array of 2d arrays
    e : array (external assets)
***



<b><code>separate_effect_measures(networks)</code></b>
    
    Apply 'network_meassures(net)' on each subnetwork, store the resutl in a comma separated string 
  

<b>type</b> : 

    string
  

<b>Parameters:</b>

    networks : array of 2d arrays
    
***

<b><code>naked_cds(j,k,contract)</code></b>
    
    Return sum of the value of notionals writen on k in which j is holder.

<b>type</b> : 

    float
  

<b>Parameters:</b>

    j : int  ( index of j)
    i : int (index of i)
    contract : 3d-array (matrix of CDS contracts) 
  
***

<b><code>is_naked(j, k, l, c)</code></b>
    
    Return 1 if j has naked position towards k; otherwise return 0


<b>type</b> : 

      int
  

<b>Parameters:</b>

      j : int  ( index of j)
      k : int (index of i)
      l : 2d-array ((debt contracts)liability matrix)
      contract : 3d-array (matrix of CDS contracts) 

***

<b><code>cds_over_debt(j, k, l, c)</code></b>
    
    Return cds over debt ratio


<b>type</b> : 

      float
  

<b>Parameters:</b>

      j : int  ( index of j)
      k : int (index of i)
      l : 2d-array ((debt contracts)liability matrix)
      contract : 3d-array (matrix of CDS contracts) 

</div>

We will use these measures in the analysis framework which we will explain it later.


   

## Random Network Generators 

 Random network generators for both uniform networks and bow tie networks are implemented in python. 
```bash
- uniform_network_generator.py
- bowtie_network_generator.py
```

The detailed description of these random network generators are provided in the report, sections 3.2.2 to 3.2.3 . 

In uniform_network_generator.py :



<div class="alert alert-block alert-info">

<b><code>uniform_network_gen (n_banks, p)</code></b>

    Return the uniform network generated given denisty (p)
    
<b>type</b> : 
    
    debt_matrix, 2d-array of debt contracts
    contract,  3d-array of cds contracts
    ref_entities, list of reference entities


<b>Parameters:</b>

    n_banks: int (number of banks, nodes, in the network)
    p : float ( density of network should be in (0, 1))

</div>


In *bowtie_network_generator.py*:

<div class="alert alert-block alert-info">

<b><code>bowtie_network_gen(n_banks, net_buyers_fraction, net_sellers_fraction, density_overall, prob_dealer_dealer)
    </code></b>

    Return the bow tie network generated given parameters
    
<b>type</b> : 
    
    debt_matrix, 2d-array of debt contracts
    contract,  3d-array of cds contracts
    ref_entities, list of reference entities


<b>Parameters:</b>

    n_banks: int (number of banks, nodes, in the network)
    net_buyers_fraction : float ( share of buyers in the network)
    net_sellers_fraction : float ( share of sellers in the network)
    density_overall : float ( density of the core)
    prob_dealer_dealer : float ( probability of link formation among dealers)

</div>


## External Assets Generators

External assets are generated as described in the report (section 3.2.3). This module is implemented in python under *extergen.py*, illustrated below:


<div class="alert alert-block alert-info">

<b><code>external_gen(li, lvr)</code></b>

    Return the array of external assets.
    
<b>type</b> : 
    
    array

<b>Parameters:</b>

    li : 2d array  ( liability matrix)
    lvr : float (leverage ratio)

</div>

Following is sample output of the method.


```python
from extergen import *

li = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 94.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 24.0, 0.0, 67.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 12.0, 0.0, 35.0, 20.0, 0.0, 0.0, 85.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [63.0, 0.0, 0.0, 0.0, 0.0, 34.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 0.0],
      [24.0, 0.0, 0.0, 67.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 75.0, 0.0, 20.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 56.0, 0.0, 63.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
      [0.0, 0.0, 0.0, 94.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0]]

lr = 0.3 # leverage ratio

print external_gen(li, lr) 

```

    [0, 0, 36.57142857142858, 0, 99.28571428571429, 87.42857142857144, 191.42857142857144, 121.57142857142857, 0, 82.00000000000001, 0, 68.71428571428572, 21.000000000000007, 78.85714285714286, 147.42857142857144]
    



## Analysis Framework

Putting all the modules together we will be able to define the analysis framework. There are two python codes for this purpose:

- cl_uni_main.py
- cl_btie_main.py

In all these code we are going to import all the modules descibed so far, that is :



```python
import bowtie_network_generator as bg
import clearing_algorithm_sim as cl
from extergen import *
import shock_gen as shck
import uniform_network_generator as ug

```

First part of these frameworks is parameters declaration, for example for cl_uni_main.py, we need to declare parameters as:



```python
## PARAMETERS

n_banks =50

alpha, beta = 0.6, 0.6 # default cost parameters
levRatio =[0.02, 0.052, 0.12] # leverage ratio
lossRate =[ 0.4, 0.6, 0.95] # loss rates

#network generator's specific parameters :

### getting the rest of parameters from console. To be used on cluster, 
#please refer to the bash_uni.py to understand the reason behind it.

p, num = float(sys.argv[1]), float(sys.argv[1])
 
'''
maximum and minimum number of iterations,

here 5 times the code is executed within each iteration

'''
min_iter, max_iter = 5*num, 5*(num+1) 

```

After defining the parameters, we can call the random network generators, it is done by :


```python
li, c, cds_banks= ug.uniform_network_gen (n_banks, p)
# generate uniform random network
```

We will generate 'id's to take track of the simulation results (See the codes, it is comprehensible). Afterwards, we will calculate the network measures as described earlier. 



```python
# calculate network measures
networks, n_naked, n_cont, n_cycles,avg_cds_debts,max_cds_debts,std_cds_debts = \
ml.convert_to_sep_networks(li, c, cds_banks)

measures_so_far = ml.separate_effect_measures(networks)+","+str(n_naked)+","+\
str(n_cont)+","+str(n_cycles)+","+str(avg_cds_debts)+","+str(max_cds_debts)\
+","+str(std_cds_debts)

```

In addition, we will also use 'frag_index' method as soon as we have externall assets.



```python
# generate external asset give leverage ratio 

ea, debts_over_assets = external_gen(li, c, leverageratio)

# get frag index: leverage of network  for different subnetworks

frag_indices = ml.frag_index(networks, ea)

fi_debt, fi_cds, fi_agg = frag_indices[0], frag_indices[1], frag_indices[2] 

avg_fi = np.mean(frag_indices)

# dos : debts over assets, it is specified in the extergen.py 

avg_dos, std_dos, max_dos = np.average(debts_over_assets), np.std(debts_over_assets), np.max(debts_over_assets)

sum_ea, avg_ea, st_ea = sum(ea), np.average(ea), np.std(ea)


```

Now for different parameter sets, we will run the compression algorithm, that is :

- For different leverage ratios, generate external assets
- For each of the vector of external assets generated in previous step,do :
    - for all loss rates, apply shock generator
- invoke the clearing algorithm (*clearing_algorithm_sim.py*)
- and Provide systemic risk measures

These steps can be summarized in the following piece of code :




```python
for leverageratio in levRatio:
    # step 1 . generating external assets given leverage ratio
    
    ea, debts_over_assets = external_gen(li, c, leverageratio)
    # default cascades algorithm on the uncompressed network 
    sol1, eq1,r1 = dc.DefaultCascades(li, ea, alpha, beta)
    
    # step 2. apply shocks
        
    for rate in lossRate:
        
         post_external_assets = np.asarray(shck.shock_gen(n_banks, ea, rate)).tolist()
            
        # step 3. clearing problem is invoked
        for l in xrange(nBanks):
        
            f_vec, total_recovery_rates, total_equity, infeasible_out,\
            maximal_out =cl.clearing_algorithm(id_output,post_external_assets, li, c, cds_banks)

            
          # step 4 . systemic risk measures are calculated ....  
```

Steps above are done for all the analysis frameworks. The are more details in the codes, such as id generation, etc.

The output is a comma seperated string, writen to a file, CSV file ( includes all the systemic risk and network measures calculated).

Following Diagram Will depict the analysis framework we have used.


<img src="framework.png" />


 

# Simulation on the Kraken Cluster


## Installation

Installation follows the stepst discussed in the previous sections, however one should note that, due to absence of priviledge rights on the cluster, it is not possible to follow steps which has the system command "sudo", so any linux commands such as "sudo apt-get isntall" could not work.

The best aproach is to install "pip" manually to handle python related installations and use anaconda to install all the rest. It is clear that all the path variables need to be set on "~/.zshrc" file, this process is going to be discussed briefly in this documentation.



## Submiting jobs to cluster

Two python files are provided, namely bash_uni.py and bash_bowtie.py . These files are responsible for submiting  jobs for the simulation to the cluster. For illustrative purposes we are going to discuss '*bash_bowtie.py*' here.



```python
import math
import sys
import os



'''
Generat bash files with the given parameter sets and submit them to the Kraken cluster



parameters can be chaned by changing the values within the for loop

these values declare the corresponding parameters of uniform_network_generator.py ( density)

output : Uniform_#n.sh

'''
#counter to keep track of files being generated throughout this code
counter = 0
data = ''
#number of banks used in the simulation
nBanks=50
cwd = os.getcwd()
for pvalue in [0.20, 0.25, 0.3, 0.4, 0.9] :
    content = "#!/bin/bash \n#!/bin/bash \n# SBATCH --job-name=clearing_opt_uni\
    \n# SBATCH --cpus-per-task=1\n# SBATCH -t 0-11:59:59\n# SBATCH --mem=MaxMemPerNode\n"
    data =data + "python "+str(cwd)+"/cl_uni_main_2.py "+str(pvalue)+" ${SLURM_ARRAY_TASK_ID}"
#    print data 
    file = open("Uniform_"+str(counter)+".sh", 'a')
    file.write(content)
    file.write(data)
    file.close()
    data = ''
    os.system('sbatch --partition="kraken_superfast" --array=0-1000 --nodes=1\
    --ntasks=1 --cpus-per-task=1 --mem-per-cpu=2 ./Uniform_2_'+str(counter)+'.sh')
    counter+=1
```

This code will generate a number of bash files and at the same time the genrated scripts will be submited to the cluster. We have used following specifications to run each job :

- partition : kraken_superfast
- job arrays : 0-1000
- nuber of nodes : 1
- number of tasks : 1
- cpus per task : 1
- mem per cpu : 2

we will keep track of the simulation in the main files by use of  

```bash

${SLURM_ARRAY_TASK_ID}

```
and within each job, we will run the simulation based on the predefined parameters, max_iter and min_iter defined in the main files. It means that for each job submited to the cluster we will run the analysis framework n times for example, this number could be set by manipulating max_iter and min_iter.

The outcoume of the simulation is :

1. different directories for each parameter set
2. in each directory there are different text files generated by the main files. Each line in a text file corresponds to a row of a CSV file. Headings of the csv files are provided in different files, that is uniform_heading.csv, bowtie_heading.csv. 

First wee will create four directories for each simulation of the corresponding random network genreators:
```bash

mkdir uniform 
mkdir bowtie

```


Whenever the simulation of one specific class of random network generators is finished, we need to concatenate all the text files to one CSV file. This could be done by running following commands.


```bash

cd /toTheOutcomeDirectory  
touch somename.csv
find . -type f -name '*.txt' -exec cat {} + >>1.csv  # use numbers for naming.
cp 1.csv /pathTo/uniform

```

This procedure should be done for each directory generated by running the main file. Forexample, for the uni_main.py we will have six directories for each network density values. After this step to agregate all the results we should do :

```bash

cd /pathTo/uniform
touch result_uniform.csv
cat uniform_heading.csv >> result_uniform.csv
# the sequence should be within the range of number of CSV files generated.
for i in `seq 1 6`; do
 cat $i.csv >> result_uniform.csv
done

tar -czvf sim_result_uniform.tar.gz result_uniform.csv

```

The compressed file should be used for analysis part. The file could be downloaded via the following command.

```bash
scp -r yourid@gru.ifi.uzh.ch:/pathTo/uniform/sim_result_uniform.tar.gz /pathToYourLocalDirecory
```

A rough estimate for the simulation time varies by the solver's parameters, but each job ( we have 1000 for each class of random network generators) will take approximately between 40-54 minutes. Below is the approximate simulation time for each class of network generators:

- uniform : 20-24 hours
- bowtie1 : 48-50 hours

In the next section we are going to describe how we can do the statistical analysis. However, this step is done locally and not on the cluster.

## SOSTOOLS

We will use SOSTOOLS (Sum of Squares Optimization Toolbox for MATLAB). "SOSTOOLS is a free, third-party MATLAB1 toolbox for solving sum of squares programs. The techniques behind it are based on the sum of squares decomposition for multivariate polynomials, which can be efficiently computed using semidefinite programming."

Here is a list of requirements for SOSTOOLS installation: 
- MATLAB R2009a or later.
- Symbolic Math Toolbox version 5.7 (which uses the MuPAD engine). 
- The following SDP solver: SeDuMi. This solver must be installed before SOSTOOLS can be used. The user is referred to the relevant documentation to see how this is done. The solvers can be downloaded from: [SeDuMi](http://sedumi.ie.lehigh.edu)

The software and its user’s manual can be downloaded from this [link](http://www.eng.ox.ac.uk/control/sostools/).
Once you download the zip ﬁle, you should extract its contents to the directory where you want to install SOSTOOLS
In **Windows** , after extracting the file, you must add the SOSTOOLS directory and its subdirectories to the MATLAB path. This completes the SOSTOOLS installation.

We will use following codes to provide certificates. This code is implemented based on Parillo 2000 and more technicall discussion of S.Prajna, 2013.
In the *'p_certificates.m'* we have provided the complete implementation of this result. The main code is also provided in the file *'certificate.m'*. One should add the files generated from the clearing_algorithm_sim.py in the matlab working directory.



```python
function [GAM,vars,xopt] = p_certificates(p,ineq,eq,DEG,options)

switch nargin
    case 1 
        options.solver='sedumi';
        ineq = [];
        eq = [];
    case 2 
        options = ineq;
        ineq = [];
        eq = [];
    case 3
        options.solver='sedumi';
        ineq = ineq(:);
        DEG = eq;
        eq = [];
    case 4
        if ~isstruct(DEG)
            ineq = ineq(:);
            eq = eq(:);
            options.solver='sedumi';
        else
            options = DEG;
            ineq = ineq(:);
            DEG=eq;
            eq = [];
        end
    case 5 
        ineq = ineq(:);
        eq = eq(:);
end


vect = [p; ineq; eq];

% Find the independent variables, check the degree
if isa(vect,'sym')
   varschar = findsym(vect);
   class(varschar)
   vars = str2sym(['[',varschar,']']);
   nvars = size(vars,2) ; 
   if nargin > 2
       degree = 2*floor(DEG/2);
       deg = zeros(length(vect),1);
       for i = 1:length(vect)
           deg(i) = double(feval(symengine,'degree',vect(i),converttochar(vars)));
           if deg(i) > degree
               error('One of the expressions has degree greater than DEG.');
           end;
       end;   
   else
       for var = 1:nvars;
           if rem(double(feval(symengine,'degree',p)),2) ;
               disp(['Degree in ' char(vars(var)) ' should be even. Otherwise the polynomial is unbounded.']);
               GAM = -Inf;
               xopt = [];
               return;
           end;
       end;
   end;
   
else
   varname = vect.var;
   vars = [];
   for i = 1:size(varname,1)
       pvar(varname{i});
       vars = [vars eval(varname{i})];
   end;
end;

% Construct other valid inequalities
if length(ineq)>1
    for i = 1:2^length(ineq)-1
        Ttemp = dec2bin(i,length(ineq));
        T(i,:) = str2num(Ttemp(:))';
    end;
    T = T(find(T*deg(2:1+length(ineq)) <= degree),:);
    T
    deg = [deg(1); T*deg(2:1+length(ineq)); deg(2+length(ineq):end)];
    for i = 1:size(T,1)
        ineqtempvect = (ineq.').^T(i,:);
        ineqtemp(i) = ineqtempvect(1);
        for j = 2:length(ineqtempvect)
            ineqtemp(i) = ineqtemp(i)*ineqtempvect(j);
        end;
    end;
    ineq = ineqtemp;
end;

% First,we need initialize the sum of squares program
prog = sosprogram(vars);
expr = -1;

for i = 1:length(ineq)
    [prog,sos] = sossosvar(prog,monomials(vars,0:floor((degree-deg(i+1))/2)));
    expr = expr - sos*ineq(i);
end;
for i = 1:length(eq)
    [prog,pol] = sospolyvar(prog,monomials(vars,0:degree-deg(i+1+length(ineq))));
    expr = expr - pol*eq(i);
end;
% Next, define SOSP constraints
prog = soseq(prog,expr);
% And call solver
[prog,info] = sossolve(prog);

if (info.dinf>1e-2)|(info.pinf>1e-2)
    disp('No certificate.Problem is infeasible');
else
    
%     a = sosgetsol(prog);
    disp('certficate exists');
         
end;




```

Besides as could be seen from the following lines of the *'certificate.m'* : in list (id_list), one need to specify all the ids generated in the clearing algorithm that corresponds to infeasible or non maximal cases. (they are kept separately in 'id_file_nonmax.txt' and 'id_file_infeasible.txt')


```python

clear; echo on;
% define symbolic variables

% 50 is number of banks, here.
% d is default  indicator in the feasiblity problem
sym('d', [1 50])
D = sym('d', [1 50]);
syms(D);

% r is recovery rate in the feasibility problem 
R = sym('r', [1 50]);
syms(R);

vartable = horzcat(D,R);

% read from mat files to keep track of polynomials generated from Clearing

% Optimization Problem

% use the list from clearing_algorithm_sim.py in here to keep track of ids
% change this list to the list descibed above.
id_list = ["2"];

for i=1:length(id_list)
  id = id_list(i);


% use following file names if you are considerig infeasibility results; otherwise, use the commented lines
polynomialsfile = load(id+'_infeasible_polynomials.mat');
%polynomialsfile = load(id+'_infeasible_polynomials.mat');

polynomials = strtrim(string(polynomialsfile.vect));

sospolynomialsfile = load(id+'_nonmax_sospolynomials.mat');
%sospolynomialsfile = load(id+'_nonmax_sospolynomials.mat');

sospolynomials = strtrim(string(sospolynomialsfile.vect));



%polyvars
m = cell(1,length(polynomials));
p = cell(1,length(polynomials)+length(sospolynomials)+1);
q = cell(1,length(sospolynomials));

for i = 1:length(polynomials)
    m{i} = str2sym(polynomials(i));
   
end

for i =1:length(sospolynomials)
    q{i} = str2sym(sospolynomials(i));
end
p_certificates(-r1, vertcat(q{length(m)/10:length(m)/10+3}),vertcat(m{2:4}),4);

end
```

The output of this code is whether the solver has found a certificate or not.

## Appendix

### zshrc file :

The zshrc file should look like :


#### Paths
```bash
    PATH="$PATH:$HOME/bin"
    export PATH=$PATH:/home/user/$USERID/ampl.linux64
    export PATH=$PATH:/home/user/$USERID/anaconda2/bin
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/user/$USERID/anaconda2/lib/
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/user/$USERID/anaconda2/lib/R/lib
```

## References

Rezakhani, Pouyan. 2019. "Analysis of Systemic Risk in Financial Networks with Credit Default Swpas via Monte Carlo Simulations", MSc Thesis. University of Zurich.

Parrilo, Pablo A. 2000. “Structured semidefinite programs and semialgebraic geometry methods in robustness and optimization.” PhD diss. California Institute of Technology.

S. Prajna, A. Papachristodoulou, P. Seiler, and P. A. Parrilo. 2013, “SOSTOOLS: Sum of squares optimization toolbox for Matlab,” Available from http://www.cds.caltech.edu/sostools

