import random


def external_gen(li, c, leverageratio):
  v = [0 for i in range(len(li))]
  r = [0 for i in range(len(li))]
  for i in range(len(li)):
    assetnotional = get_ingoing(li, c, i)
    liabilitynotional = get_outgoing(li, c, i)
    r[i] = liabilitynotional/(assetnotional+0.00000000000000000000000000000001)
    diff = liabilitynotional-(1-leverageratio)*assetnotional
    if (diff>0):
      v[i]=diff/(1-leverageratio)

  return v, r  


def get_ingoing(li, c, i):
	sum = 0
	for j in range(len(li)):
		sum+=li[j][i]
		# for k in range(len(li)):
		# 	if c[j][i][k]!=0:
		# 		sum+=c[j][i][k]

	return sum		

def get_outgoing(li, c, i):
	sum = 0
	for j in range(len(li)):
		sum+=li[i][j]

		# for k in range(len(li)):
			# if c[i][j][k]!=0:
				# sum += c[i][j][k]


	return sum		

