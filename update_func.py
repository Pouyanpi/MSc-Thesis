import sys
import ast


negligible = 0.000000000000000000000000001

def cds_debt(i,j,reference_entities,recovery_rates,cds_contract):
	sum = 0
	for k in reference_entities:

		sum = sum + (1-recovery_rates[k])*cds_contract[i][j][k]
	return sum	

def liabilities(n_banks,reference_entities,debt_contract,cds_contract,recovery_rates):

	liability = [[0 for k in xrange(n_banks)] for j in xrange(n_banks)]

	for i in xrange(n_banks):
		for j in xrange(n_banks):
			sum = cds_debt(i,j,reference_entities,recovery_rates,cds_contract)
			liability[i][j] = debt_contract[i][j] + sum

	return liability

def total_liability(n_banks,liability):
	
	total_li = [0 for i in range(n_banks)]

	for i in xrange(n_banks):
		
		total_li[i] = sum(liability[i][:])


	return total_li	
	
def payment(n_banks,liability, recovery_rates):
	
	payments = [[0 for k in xrange(n_banks)] for j in xrange(n_banks)]
	
	for i in xrange(n_banks):
		for j in xrange(n_banks):
	
			payments[i][j]=(liability[i][j]*recovery_rates[i])	

	return payments
def interbank_assets(i, n_banks, payments):
	result =0
	for j in xrange(n_banks):
		result = result + payments[j][i]
	return result	

def total_assets(n_banks,payments,ea,total_li, alpha, beta):
	total_assets = [0 for i in range(n_banks)]

	for i in xrange(n_banks):
		summ = interbank_assets(i,n_banks,payments)
	
		if (ea[i]+summ < total_li[i]):
	
			total_assets[i]= beta*summ+alpha*float(ea[i])
	
		else:
	
			total_assets[i] = summ+float(ea[i])	

	return total_assets	

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

