
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