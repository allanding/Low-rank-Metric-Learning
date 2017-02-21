function testJianFang()
d=10000;
M=100;
n=100;
X=rand(d,n);
Q=orth(rand(d,M));
W=rand(M,2);
Y=[ones(1,44),-ones(1,n-44)];
Y2=[Y;-Y];

opts.record = 0;
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
out.tau = 1e-3;

tic; [Q, out,objs]= OptStiefelGBB(Q, @fun_eval, opts, X,W,Y2); tsolve= toc
nnz(2*((W'*Q')*X>0)-1==Y2)
plotLine(objs(2:end))
%1
function [F G]=fun_eval(Q,X,W,Y)
obj=(W'*Q')*X-Y;
F=norm(obj,'fro');
G=X*(obj'*W');

