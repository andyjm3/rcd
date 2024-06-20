function [X, infos] = coordinate_descent_symplectic(problem, X0, options)
    
    sym = @(A) 0.5*(A + A');

    [n p] = size(X0);
    n = n/2;
    p = p/2;

    idx1 = [];
    idx2 = [];

    for ii = 1 : 2*n
        for jj = ii+1 : 2*n
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

        if strcmp(update_type, 'replace')
            seqlist = randsample(nn, numupdates, true);
        elseif strcmp(update_type, 'noreplace') 
            seqlist = randperm(nn, min(nn,numupdates)); % max of nn coordinates  
        elseif strcmp(update_type, 'cyclic')
            seqlist = 1:min(nn,numupdates); % max of nn coordinates  
        end
        

        % Stepsize sequence
        stepsize = stepsize0;

        t0 = tic();

        % Compute egrad once.
        if linearization
            egrad = problem.egrad(X); % problem dependent. size is 2n by 2p.
        end
    
        
        % Update the coordinates
        for k = 1: length(seqlist)

            i = idx1(seqlist(k));
            j = idx2(seqlist(k));


            % ith row of OmegaX is i1th row of X with signi1. 
            % Similarly for jth row.
            if i <= n
                i1 = n + i;
                signi1 = 1;
            else
                i1 = i - n;
                signi1 = -1;
            end

            if j <= n
                j1 = n + j;
                signj1 = 1;
            else
                j1 = j - n;
                signj1 = -1;
            end

         
            if linearization
                gradi = egrad(i,:); 
                gradj = egrad(j,:); 
            else
                gradi = problem.egradrowi(X, i); 
                gradj = problem.egradrowi(X, j);
            end

            % Compute etaij

            etaij = signi1 * (  X(i1,:) * gradj'  )   +    signj1 * (  gradi * X(j1,:)'  ); % 4p
            etaij = - stepsize * etaij;


            if i == j - n 
                X(i, :) = exp(-etaij) * X(i, :);
                X(j, :) = exp(etaij) * X(j, :);

            else
                tempXi = X(i, :) + (etaij*signj1) * X(j1, :); % 2p
                tempXj = X(j, :) + (etaij*signi1) * X(i1, :); % 2p 

                X(i, :) = tempXi;
                X(j, :) = tempXj;

            end


        end


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
    
    infos.flops = [0; 8*p*length(seqlist)*ones(it,1)]; % BM: it should be 8p. 
    infos.inneriter = length(seqlist);

    infos.time = cumsum(infos.time);

end

