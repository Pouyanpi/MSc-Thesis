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



