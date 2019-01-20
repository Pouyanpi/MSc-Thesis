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
    content = "#!/bin/bash \n#!/bin/bash \n# SBATCH --job-name=clearing_opt_uni\n# SBATCH --cpus-per-task=1\n# SBATCH -t 0-11:59:59\n# SBATCH --mem=MaxMemPerNode\n"
    data =data + "python "+str(cwd)+"/cl_uni_main_2.py "+str(pvalue)+" ${SLURM_ARRAY_TASK_ID}"
#    print data 
    file = open("Uniform_"+str(counter)+".sh", 'a')
    file.write(content)
    file.write(data)
    file.close()
    data = ''
    os.system('sbatch --partition="kraken_superfast" --array=0-1000 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=2 ./Uniform_2_'+str(counter)+'.sh')
    counter+=1
