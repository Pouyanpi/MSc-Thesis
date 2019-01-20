import numpy as np
import scipy.io as sio 
import random
import scipy.io as sio
import random
import uniform_network_generator as ug
import extergen as ex

BIGM =1000000000000
def generate_polynomials(status,id,ea, li, c, cds_banks, alpha, beta):


	
	n = len(ea)
	cds_poly_var = ""
	interbank_poly_var =""
	poly_var = ""
	vect_polynomials = []
	vect_sospolynomials = []
	vartable = []
	syms = ""

	for i in xrange(n):
		total_li = str(sum(li[i][:]))
				
		for j in xrange(n):
			for k in cds_banks:
				if c[j][i][k] !=0 :
				# interbank_asset = 
					cds_poly_var = "(1-r"+str(k+1)+")*"+str(c[j][i][k])
				else :
					cds_poly_var = "0"	
			if li[j][i] != 0:
			
				interbank_poly_var = "r"+str(j+1)+"*("+str(li[j][i])+"+"+cds_poly_var+")"
			elif cds_poly_var!="0" : 
				interbank_poly_var = "r"+str(j+1)+"*("+cds_poly_var+")"
			else :
				interbank_poly_var = "0"
					
		if total_li !="0.0":		

			poly_var = "r"+str(i+1)+"*"+total_li+"-"+"d"+str(i+1)+"*"+total_li+"-"+"(1-d"+str(i+1)+")*("+str(alpha*ea[i])+"+"+str(beta)+"*("+interbank_poly_var+"))"

		poly_var = "-"+"(1-d"+str(i+1)+")*("+str(alpha*ea[i])+"+"+str(beta)+"*("+interbank_poly_var+"))"
		vect_polynomials.append(poly_var)
		vect_polynomials.append("d"+str(i+1)+"*(d"+str(i+1)+"-1)")
		vect_sospolynomials.append("("+str(ea[i])+"+"+interbank_poly_var+"-"+total_li+")-"+str(BIGM)+"*d"+str(i))
		vect_sospolynomials.append("r"+str(i+1)+"*(1-r"+str(i+1)+")")
		vartable.append("d"+str(i+1))
		vartable.append("r"+str(i+1))
		syms += "d"+str(i+1)+" "+"r"+str(i+1)+" "
		
		if status =="infeasible":


			file_m1 = id+'_infeasible_polynomials.mat'
			file_m2 = id + '_infeasible_sospolynomials.mat'
		
			sio.savemat(file_m1, {'vect': vect_polynomials})
			sio.savemat(file_m2, {'vect': vect_sospolynomials})
		elif status == "nonmax":
			file_m1 = id+'_nonmax_polynomials.mat'
			file_m2 = id + '_nonmax_sospolynomials.mat'
		
			sio.savemat(file_m1, {'vect': vect_polynomials})
			sio.savemat(file_m2, {'vect': vect_sospolynomials})
				
	return vect_polynomials, vect_sospolynomials, vartable, syms	



nbanks =10
id = "2"


	

# li, c, cds_banks = ug.uniform_network_gen (10, 0.2)
# ea,b = ex.external_gen(li, c, 0.052)
# alpha,beta =0.6,0.6
# vp, vsosp, vt, syms = generate_polynomials(id, ea, li, c , cds_banks, alpha, beta)
