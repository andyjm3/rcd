function [X, infos] = RGD_sp(problem, X0, options)
    [n p] = size(X0);
    n = n/2;
    p = p/2;

    M = problem.M;

    localdefaults.maxiter = 100;
    localdefaults.stepsize = 1;
    localdefaults.method = 'QG'; 
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
        U = M.egrad2rgrad(X, U); % 40 np^2 + 8p^3 + 8p^2 + 4np

        if strcmpi(method, 'QG') 
            X = M.retr_qg(X, U, -stepsize); 
        elseif strcmpi(method, 'CL') 
            X = M.retr_cl(X, U, -stepsize);
        elseif strcmpi(method, 'SR') 
            X = M.retr_sr(X, U, -stepsize);
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
    
    flops_rgrad = 40 *n*p^2 + 8*p^3 + 8*p^2 + 4*n*p;

    flops_qg = flops_rgrad + 64 * n*(p^2) + 104 *(p^3) +  28*(p^2) + 6 *n*p;
    flops_cl = flops_rgrad + 64 * n*(p^2) + 8 *(p^3) + 16 * (p^2) + 12*n*p;
    flops_sr = flops_rgrad + 48 * n*(p^2) ;



    if strcmpi(method, 'QG') 
        infos.flops_qg = [0; flops_qg*ones(it,1)];
    elseif strcmpi(method, 'CL')
        infos.flops_cl = [0; flops_cl*ones(it,1)];
    elseif strcmpi(method, 'SR')
        infos.flops_sr = [0; flops_sr*ones(it,1)];
    end


    infos.time = cumsum(infos.time);
end

