function[X, infos] = ExPen(problem, X0, options)

	[n p] = size(X0);

    sym = @(P) (P + P')/2;

    localdefaults.maxiter = 100;
    localdefaults.method = 'SD'; 
    localdefaults.tolgradnorm = eps;
    localdefaults.maxtime = inf;
    localdefaults.lam = 1;
    localdefaults.minoptgaptol = 1e-3;


    options = mergeOptions(localdefaults, options);

    method = options.method;
    lam = options.lam;

    eyep = eye(p);

    function f = EPcost(Y)
        YtY = Y'*Y;
        dist = YtY - eyep;
        temp = Y * (1.5.*eyep - 0.5.*YtY);
        f = problem.cost_infea(temp) + (options.lam/4) * norm(dist, 'fro')^2;
    end

    function g = EPegrad(Y) % 9np^2+ 6p^2 + np 
        YtY = Y'*Y;    % np^2    
        temp1 = 1.5.*eyep - 0.5.*YtY; % 3p^2
        temp = Y * temp1; % 2 np^2
        GX = problem.egrad_infea(temp); % problem dep
        g = GX * temp1  +  Y * (-sym(Y'*GX) +  options.lam *(YtY-eyep)); % 6np^2 + 3p^2 + np
    end


    problemEP.M = euclideanfactory(n,p);
  	problemEP.cost = @EPcost;
    problemEP.egrad = @EPegrad;

    

    function stats = mystatsfun(problemEP, X, stats)
        stats.mycost = problem.cost(X);
        stats.mygradnorm = problem.gradnorm(X);
        stats.myoptgap = problem.optgap(X);
    end

    function stopnow = mystopfun(problemEP, X, info, last)
        stopnow = problem.optgap(X) < options.minoptgaptol;
    end


    if strcmp(method, 'SD') 
        localdefaults.stepsize = 1;
        options = mergeOptions(localdefaults, options);

        maxiter = options.maxiter;   
        stepsize0 = options.stepsize;
        M = problemEP.M;
        infos.time = nan(maxiter,1);
        infos.cost = nan(maxiter, 1);
        infos.gradnorm = nan(maxiter,1);
        infos.optgap = nan(maxiter,1);


        X = X0;

        for it = 1:maxiter

            % Stepsize sequence
            stepsize = stepsize0;

            t0 = tic();
            
            U = problemEP.egrad(X); % 9np^2+ 6p^2 + np 
            X = X - stepsize* U;% 2np
                            
            timeperit = toc(t0);

            infos.time(it, 1) = timeperit;
            infos.cost(it, 1) = problem.cost(X);
            infos.gradnorm(it, 1) = problem.gradnorm(X);
            infos.optgap(it, 1) = problem.optgap(X);

            fprintf('%s:  %3d\t%+.3e \t%.3e\n', 'ExPen-SD', it, infos.optgap(it), stepsize);

            if sum(infos.time(1:it, 1)) > options.maxtime || infos.optgap(it, 1) < options.minoptgaptol
                break;
            end

        end
        infos.time = [0; infos.time(1:it,1)];
        infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
        infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
        infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];
    
        sdflops = 9*n*(p^2) + 6*(p^2) + 3*n*p;
        infos.flops_sd = [0; sdflops*ones(it,1)];


        infos.time = cumsum(infos.time);

    elseif strcmp(method, 'CG')
        
        options.statsfun = @mystatsfun;
        options.stopfun = @mystopfun;

        [X, ~, infos_CG, options] = conjugategradient(problemEP, X0, options);

        infos.time = [infos_CG.time];
        infos.cost = [infos_CG.mycost];
        infos.gradnorm =[infos_CG.mygradnorm];
        infos.optgap = [infos_CG.myoptgap];
        
    end

    X = qr_unique(X);
end