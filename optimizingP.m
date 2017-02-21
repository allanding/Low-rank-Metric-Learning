function P = optimizingP(P,Y,Lw,X,Q,Z,n,la0,E,Y1,Y3,mu,maxIter)
  opts.record = 0;
  opts.mxitr  = maxIter;
  opts.xtol = 1e-3;
  opts.gtol = 1e-3;
  opts.ftol = 1e-4;
  opts.tau = 1e-3;

  %% optimize P by call objHereInnner

  P=OptStiefelGBB(P, @objHereInner, opts, X,Y,Lw,Z,E,Q,Y1,Y3,n,la0,mu);
end
