function [X, infos] = landing(problem, X0, options)
    [n p] = size(X0);

    localdefaults.maxiter = 100;
    localdefaults.stepsize = 1;
    localdefaults.safe_stepsize = true;
    localdefaults.lam = 1;
    localdefaults.eps = 0.5;
    localdefaults.maxtime = inf;
    localdefaults.minoptgaptol = 1e-3;



    options = mergeOptions(localdefaults, options);


    maxiter = options.maxiter;   
    
    infos.time = nan(maxiter,1);
    infos.cost = nan(maxiter, 1);
    infos.gradnorm = nan(maxiter,1);
    infos.optgap = nan(maxiter,1);


    X = X0;

    reg_ld = 1e-7; % avoid division by small number
   
    for it = 1:maxiter

        t0 = tic();
        
        egradst = problem.egrad_infea(X);
        XtX = X'*X; % np^2
        relgradX = (egradst * XtX - X * (egradst' *X))/2; % 6np^2 + 2np
        distX = X * XtX - X; % 2 np^2 + np
        dir = relgradX + options.lam * distX; % 2 np


        if options.safe_stepsize % total flops: 2 p^2 + 4np
            dd = sqrt(norm(XtX,'fro')^2   + n - 2*norm(X,'fro')^2) ;%  norm(dist, 'fro'); % 2p^2 + 2 np
            gg = norm(dir, 'fro'); % 2np
            
            %{
            a = norm(relgrad, 'fro');
            alpha = 2 * (params.lam * dd - a * dd - 2 * params.lam * dd);
            beta = a^2 + params.lam^2 * dd^3 + 2 * params.lam * a * dd^2 + a^2 * dd;
            max_stepsize = (alpha + sqrt(alpha^2 + 4 * beta * (params.eps - dd))) / 2 / beta;
            stepsize_ld = min(params.stepsize,max_stepsize);
            %}
            
            beta = options.lam * dd * (1 - dd);
            alpha = gg^2;
            tmp1 = max(alpha * (options.eps - dd), 0);
            sol = (beta + sqrt(beta^2 + tmp1)) / (alpha + reg_ld);
            max_stepsize = min(sol, 1 / (2 * options.lam) );
            stepsize_ld = min(options.stepsize, max_stepsize);
            
        else
            stepsize_ld = options.stepsize;
        end

        X = X - stepsize_ld .* dir;  % 2 np

        % total flops: safe_stepsize: 9np^2 + 2 p^2 + 11 np
        % total flops: fix: 9np^2  + 7 np

                        
        timeperit = toc(t0);

        infos.time(it, 1) = timeperit;
        infos.cost(it, 1) = problem.cost(X);
        infos.gradnorm(it, 1) = problem.gradnorm(X);
        infos.optgap(it, 1) = problem.optgap(X);


        fprintf('%s:  %3d\t%+.3e \t%.3e\n', 'Landing', it, infos.optgap(it), stepsize_ld);

        if sum(infos.time(1:it, 1)) > options.maxtime || infos.optgap(it, 1) < options.minoptgaptol
            break;
        end
    end

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];

    safeflops = 9*n*(p^2) + 2*(p^2) + 11*n*p;
    fixflops = 9*n*(p^2) + 7*n*p;
    

    if options.safe_stepsize
        infos.flops_safe = [0; safeflops*ones(it,1)];
    else
        infos.flops_fix = [0; fixflops*ones(it,1)];
    end

    


    infos.time = cumsum(infos.time);

    X = qr_unique(X);
end