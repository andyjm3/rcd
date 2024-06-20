function [X, infos] = RGD(problem, X0, options)
    [n p] = size(X0);
    M = problem.M;

    sym = @(P) (P + P')/2;

    localdefaults.maxiter = 100;
    localdefaults.stepsize = 1;
    localdefaults.method = 'QR'; 
    localdefaults.maxtime = inf;
    localdefaults.minoptgaptol = 1e-3;


    options = mergeOptions(localdefaults, options);


    maxiter = options.maxiter;   
    stepsize0 = options.stepsize;
    method = options.method;
    
    infos.time = nan(maxiter,1);
    infos.cost = nan(maxiter, 1);
    infos.gradnorm = nan(maxiter,1);
    infos.optgap = nan(maxiter,1);


    X = X0;
   
    for it = 1:maxiter

        % Stepsize sequence
        stepsize = stepsize0;

        t0 = tic();
        
        U = problem.egrad(X);

        if strcmpi(method, 'QR') % 6 np^2 + 3 np + 2 p^2
            U = M.egrad2rgrad(X, U);
            X = M.retr(X, U, -stepsize);
        elseif strcmpi(method, 'CL') % 7 np^2 + 4np +8 p^3
            X = M.retr_cl(X, U, -stepsize);
        elseif strcmpi(method, 'EXPF')% 6n^3 + 4 n^2p + 2n^2
            X = M.exp_large(X, U, -stepsize);
        elseif strcmpi(method, 'EXPL') % 17np^2 + 3np + 9p^3 + 2p^2
            U = M.egrad2rgrad(X, U);
            X = M.exp(X, U, -stepsize);
        end
                        
        timeperit = toc(t0);
        infos.time(it, 1) = timeperit;
        infos.cost(it, 1) = problem.cost(X);
        infos.gradnorm(it, 1) = problem.gradnorm(X);
        infos.optgap(it, 1) = problem.optgap(X);


        fprintf('%s:  %3d\t%+.3e \t%.3e\n', method, it, infos.optgap(it), stepsize);

        
        if sum(infos.time(1:it, 1)) > options.maxtime || infos.optgap(it, 1) < options.minoptgaptol
            break;
        end
    end

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];

    QRflops = 6*n*(p^2) + 3*n*p + 2*p^2;
    CLflops = 7*n*(p^2) + 8*(p^3) + 4*n*p;
    EXPflops = 6*(n^3) + 4 *(n^2)*p + 2*n^2;
    EXPlflops = 17*n*(p^2) + 9*(p^3) + 3*n*p + 2*p^2;

    if strcmpi(method, 'QR')
        infos.flops_qr = [0; QRflops*ones(it,1)];
    elseif strcmpi(method, 'CL')   
        infos.flops_cl = [0; CLflops*ones(it,1)];
    elseif strcmpi(method, 'EXPF')
        infos.flops_exph = [0; EXPflops*ones(it,1)];
    elseif strcmpi(method, 'EXPL')
        infos.flops_expl = [0; EXPlflops*ones(it,1)];
    end

    
    

    infos.time = cumsum(infos.time);
end