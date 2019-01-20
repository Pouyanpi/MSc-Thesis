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

maximize Tot_recov : total_equity;

