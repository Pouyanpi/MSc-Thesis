import math
import sys
import os


'''
Generat bash files with the given parameter sets and submit them to the Kraken cluster



parameters can be chaned by changing the values within the for loop

these values declare the corresponding parameters of bowtie_network_generator.py

output : Bowtie_#n.sh

'''
#counter to keep track of files being generated throughout this code
counter = 0
data = ''

#number of banks used in the simulation
nBanks=50
cwd = os.getcwd()

for netbuyerfraction in [0.15, 0.3, 0.9] :
        for netsellerfraction in [0.15, 0.34, 0.9]:
                for pdd in [0.1, 0.32, 0.9] :
                         
                         #calculation based on the BowtieRandomNetworkGenera
                        netbuyers = math.floor(nBanks * netbuyerfraction)
                        netsellers=math.floor(nBanks * netsellerfraction)
                        total_netbuyers_netsellers=netbuyers+netsellers
                        dealers=nBanks-total_netbuyers_netsellers
         
                        #dealers >1 are irrelavant, look at the bowtie_network_generator.py   
         
                        if dealers >1:
                            content = "#!/bin/bash \n#!/bin/bash \n# SBATCH --job-name=clearing_opt_btie\n# SBATCH --cpus-per-task=1\n# SBATCH -t 0-11:59:59\n# SBATCH --mem=MaxMemPerNode\n"
                            data =data + "python "+str(cwd)+"/cl_btie_main.py "+str(netbuyerfraction)+" "+str(netsellerfraction)+" "+str(pdd)+" ${SLURM_ARRAY_TASK_ID}"
                        #    print data 
                            file = open("Bowtie_"+str(counter)+".sh", 'a')
                            file.write(content)
                            file.write(data)
                            file.close()
                            data = ''
                            os.system('sbatch --partition="kraken_superfast" --array=0-1000 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=2 ./Bowtie_'+str(counter)+'.sh')
                            counter+=1
