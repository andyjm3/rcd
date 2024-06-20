function [X, infos] = coordinate_descent_spsd(problem, X0, options)

    [n p] = size(X0);
   
    nn = n*p;

    localdefaults.maxiter = 100;
    localdefaults.numupdates = nn;
    localdefaults.stepsize = 1;
    localdefaults.update_type = 'cyclic'; % 'cyclic', replace, noreplace
    localdefaults.maxtime = inf;
    localdefaults.minoptgaptol = 1e-3;

    options = mergeOptions(localdefaults, options);

    maxiter = options.maxiter;
    numupdates = options.numupdates;
    stepsize0 = options.stepsize;
    update_type = options.update_type;

    infos.time = nan(maxiter,1);
    infos.cost = nan(maxiter, 1);
    infos.gradnorm = nan(maxiter,1);
    infos.optgap = nan(maxiter,1);

    X = X0;


    for it = 1:maxiter

        % Stepsize sequence
        stepsize = stepsize0;

        t0 = tic();

        % Compute egrad once.
        % AH: this can be simplified according to selected coordinate
        egrad = problem.egrad(X); % problem dependent. 

        t1 = tic();
        if strcmp(update_type, 'replace')
            seqlist = randsample(nn, min(nn,numupdates), true);
        elseif strcmp(update_type, 'noreplace') 
            seqlist = randperm(nn, min(nn,numupdates)); % max of nn coordinates  
        elseif strcmp(update_type, 'cyclic')
            seqlist = 1:min(nn,numupdates); % max of nn coordinates  
        elseif strcmp(update_type, 'GS')
            [~,seqlist] = sort(egrad(:), 'descend');
            seqlist = seqlist(1:min(nn,numupdates));
        end
        tignore = toc(t1);

        % update is limited to nxp entries
        X(seqlist) = X(seqlist) -stepsize.*egrad(seqlist);  % 2*min(nn,numupdates)

        timeperit = toc(t0) - tignore;
        infos.time(it, 1) = timeperit;
        infos.cost(it, 1) = problem.cost(X);
        infos.gradnorm(it, 1) = problem.gradnorm(X);
        infos.optgap(it, 1) = problem.optgap(X);

        fprintf('%s:  %3d\t%+.3e \t%.3e\n', 'CD', it, infos.optgap(it), stepsize);

        if (sum(infos.time(1:it, 1)) > options.maxtime ) || infos.optgap(it, 1) < options.minoptgaptol
            break;
        end

    end

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];
    infos.flops = [0; 2*length(seqlist)*ones(it,1)];
    
    infos.time = cumsum(infos.time);

end

