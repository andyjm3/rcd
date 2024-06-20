function [X, infos] = RGD_spsd(problem, X0, options)
    % RGD for SPSD with AI, BW and YY geometries
    % AI BW only available in full rank
    [n, p] = size(X0);

    localdefaults.maxiter = 100;
    localdefaults.stepsize = 1;
    localdefaults.method = 'YY'; % AI, BW, YY 
    localdefaults.maxtime = inf;
    localdefaults.minoptgaptol = 1e-3;

    options = mergeOptions(localdefaults, options);

    maxiter = options.maxiter;   
    stepsize0 = options.stepsize;
    method = options.method;

    if strcmp(method, 'AI') 
        assert(n==p);
        mfd = sympositivedefinitefactory(n);
        problem.M = mfd;
    elseif strcmp(method, 'BW')
        assert(n==p);
        mfd = sympositivedefiniteBWfactory(n);
        problem.M = mfd;
    elseif strcmp(method, 'YY')
        mfd = symfixedrankYYfactory(n,p);
        problem.M = mfd;
    end

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
        U = mfd.egrad2rgrad(X, U); % 4n^3 for AI, 2n^3 + 2n^2 for bw, 0 for yy
        X = mfd.exp(X, -stepsize* U);
        % AI: n^3 + 2n^3 + n^3 + 2n^3 = 6n^3 + n^2
        % BW: 2n^2 + n^3 + 4n^3 = 2n^2 + 5n^3 + n^2
        % YY: 2n*p
        
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
    flops_ai = 4*n^3 + 6*n^3 + n^2;
    flops_bw = 2*n^3 + 2*n^2 + 3*n^2 + 5*n^3;
    flops_yy = 2*n*p;

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];
    infos.flops_ai = [0; flops_ai*ones(it,1)];
    infos.flops_bw = [0; flops_bw*ones(it,1)];
    infos.flops_yy = [0; flops_yy*ones(it,1)];

    infos.time = cumsum(infos.time);

end

