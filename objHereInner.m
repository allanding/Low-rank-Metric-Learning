function [ff, ffP]=objHereInner(P,X,Y,Lw,Z,E,Q,Y1,Y3,n,la0,mu)
    %% objective function related to P
    G0 = P'*Lw*P;
    G1 = P'*X-Y*Z-E+Y1/mu;
    G2 = P-Q+Y3/mu;
    ff=la0*trace(G0)+mu*sum(sum(G1.^2))+mu*sum(sum(G2.^2));

    %% gradient function related to P
    P1 = 2*la0*Lw + mu*X*X' + mu*eye(n);
    P2 = mu*X*(Y*Z-E)' - X*Y1' + mu*Q - Y3;
    ffP = P1*P-P2;
end
