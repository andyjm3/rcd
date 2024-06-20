function [X, infos] = coordinate_descent_orthogonal(problem, X0, options)

	[n p] = size(X0);

    idx1 = [];
    idx2 = [];
    for ii = 1 : n
        for jj = ii+1 : n
            idx1 = [idx1 ii];
            idx2 = [idx2 jj];
        end
    end
    nn = length(idx1);

    localdefaults.maxiter = 100;
    localdefaults.numupdates = nn;
    localdefaults.stepsize = 1;
    localdefaults.update_type = 'cyclic'; % 'cyclic', replace, noreplace
    localdefaults.linearization = true; %
    localdefaults.maxtime = inf;
    localdefaults.minoptgaptol = 1e-3;

    options = mergeOptions(localdefaults, options);

    maxiter = options.maxiter;
    numupdates = options.numupdates;
    stepsize0 = options.stepsize;
    update_type = options.update_type;
    linearization = options.linearization;

    infos.time = nan(maxiter,1);
    infos.cost = nan(maxiter, 1);
    infos.gradnorm = nan(maxiter,1);
    infos.optgap = nan(maxiter,1);

    X = X0;

    for it = 1:maxiter

        if strcmpi(update_type, 'replace')
            seqlist = randsample(nn, numupdates, true);
        elseif strcmpi(update_type, 'noreplace') 
            seqlist = randperm(nn, min(nn,numupdates)); % max of nn coordinates  
        elseif strcmpi(update_type, 'cyclic')
            seqlist = 1:min(nn,numupdates); % max of nn coordinates  
        end

        % Stepsize sequence
        stepsize = stepsize0;

        t0 = tic();

        % Compute egrad once.
        if linearization
            egrad = problem.egrad(X);
        end

        for k = 1: length(seqlist)
  
            i = idx1(seqlist(k));
            j = idx2(seqlist(k));

            if linearization
                gradi = egrad(i,:); 
                gradj = egrad(j,:);
            else
                gradi = problem.egradrowi(X, i); 
                gradj = problem.egradrowi(X, j);  
            end

            UXtij = gradi * X(j,:)'; % 2p
            UXtji = gradj * X(i,:)';% 2 p

            eta = - stepsize * (UXtij - UXtji);

            Xtempi = X(i,:) .* cos(eta) + X(j,:) .* sin(eta); % 3p
            Xtempj = -X(i,:) .* sin(eta) + X(j,:) .* cos(eta); % 3p

            X(i,:) = Xtempi;
            X(j,:) = Xtempj;

        end
        % 10p
        timeperit = toc(t0);
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
    
    infos.flops = [0; 10*p*length(seqlist)*ones(it,1)];
    infos.inneriter = length(seqlist);

    infos.time = cumsum(infos.time);


end