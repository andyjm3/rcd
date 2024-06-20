function [X, infos] = coordinate_descent_stiefel(problem, X0, options)
    % TSD for stiefel
    
    [n p] = size(X0);
    
    idx1 = [];
    idx2 = [];
    for ii = 1 : p
        for jj = ii : p
            idx1 = [idx1 ii];
            idx2 = [idx2 jj];
        end
    end
    nn = length(idx1);
    assert(nn == p*(p+1)/2);
    
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
            egrad = problem.egrad(X);
        end
        
        flops = 0;
        for k = 1: length(seqlist)
  
            i = idx1(seqlist(k));
            j = idx2(seqlist(k));
            
            if i ~= j
                if linearization
                    gradi = egrad(:,i); 
                    gradj = egrad(:,j); 
                else
                    gradi = problem.egradcoli(X, i); 
                    gradj = problem.egradcoli(X, j);
                end
    
                UXtij = X(:,i)' * gradj; % 2n
                UXtji = gradi' * X(:,j); % 2n

                eta = -stepsize * (UXtij - UXtji);

                Xtempi = X(:,i) .* cos(eta) - X(:,j) .* sin(eta); % 3n
                Xtempj = X(:,i) .* sin(eta) + X(:,j) .* cos(eta); % 3n

                X(:,i) = Xtempi;
                X(:,j) = Xtempj;

                flops = flops + 10*n;
            else
                if linearization
                    gradi = egrad(:,i); 
                else
                    gradi = problem.egradcoli(X, i); 
                end
                pgrad = gradi - X*(X' * gradi); % 4np + n
                dir =  -stepsize *pgrad; 
                npgrad = norm(dir); % 2n
                X(:, i) = cos(npgrad).*X(:,i) + sin(npgrad).*dir/(npgrad); % 3n

                flops = flops + 4*n*p+7*n;
            end
            
           
        end
        timeperit = toc(t0);
        infos.time(it, 1) = timeperit;
        infos.cost(it, 1) = problem.cost(X);
        infos.gradnorm(it, 1) = problem.gradnorm(X);
        infos.optgap(it, 1) = problem.optgap(X);
        infos.flops(it, 1) = flops;


        fprintf('%s:  %3d\t%+.3e \t%.3e\n', 'CD', it, infos.optgap(it), stepsize);

        if (sum(infos.time(1:it, 1)) > options.maxtime ) || infos.optgap(it, 1) < options.minoptgaptol
            break;
        end

    end

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];
    
    infos.flops = [0; infos.flops(1:it, 1)];
    infos.inneriter = length(seqlist);

    infos.time = cumsum(infos.time);

end

