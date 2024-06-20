function [X, infos] = coordinate_descent_symplectic_block(problem, X0, options)
    
    sym = @(A) 0.5*(A + A');

    [n p] = size(X0);
    n = n/2;
    p = p/2;

    idx1 = [];
    idx2 = [];
    for ii = 1 : n
        for jj = 1 : n
            if ii ~= jj
                idx1 = [idx1 ii];
                idx2 = [idx2 jj];
            end
        end
    end
    nn = length(idx1);


    localdefaults.maxiter = 100;
    localdefaults.numupdates = nn;
    localdefaults.stepsize = 1;
    localdefaults.update_type = 'cyclic'; % 'cyclic', replace, noreplace
    localdefaults.maxtime = inf;
    localdefaults.minoptgaptol = 1e-5; 
    localdefaults.linearization = 0; % whether not to call egrad after every block update

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
        egrad = problem.egrad(X); % problem dependent. size is 2n by 2p.
        symGXJ = 2*sym([X(n+1:end,:); -X(1:n,:)] * egrad'); % size is 2n by 2n. This is eta.  2*(8n^2p) + 4 n^2

        % update first block
        A = symGXJ(1:n, 1:n);
        Aupdate = A * X(n+1:end, :); %[A*X(n+1:end,1:p) A*X(n+1:end,p+1:end)]; % 4 n^2 p
        X(1:n,:) = X(1:n,:) -stepsize.* Aupdate;   % 4 np     

        if ~options.linearization
            egrad = problem.egrad(X); % problem dependent. size is 2n by 2p.
            symGXJ = 2*sym([X(n+1:end,:); -X(1:n,:)] * egrad');
        end

        % update the fourth block
        B = symGXJ(n+1:end, n+1:end);
        Bupdate = -B*X(1:n,:);%[-B*X(1:n,1:p) -B*X(1:n,p+1:end)]; % 4 n^2 p
        X(n+1:end,:) = X(n+1:end,:) -stepsize.* Bupdate; % 4 np 
        
        %{
        if ~options.linearize
            egrad = problem.egrad(X); % problem dependent. size is 2n by 2p.
            symGXJ = 2*sym([X(n+1:end,:); -X(1:n,:)] * egrad');
        end

        % update the off-diags
        u = diag(symGXJ,n);
        v = u;
        expu = exp(stepsize*u);
        expv = exp(-stepsize*v);
        expuv = [expu; expv];
        X = expuv .* X;  % 4 np
        %}

        if ~options.linearization
            egrad = problem.egrad(X); % problem dependent. size is 2n by 2p.
            symGXJ = 2*sym([X(n+1:end,:); -X(1:n,:)] * egrad');
        end
        
        %{
        % update the other coordinates
        % method 1:
        for k = 1: numupdates
            % TODO: (problematic with the update)
            i = idx1(seqlist(k));
            j = idx2(seqlist(k));

            %JX = [X(n+1:end,:); -X(1:n,:)];

            symGXJij = symGXJ(i, n+j);

            Xuupdatei = -symGXJij.* X(j,:);
            Xdupdatej = symGXJij.*X(n+i,:);

            X(i,:) = X(i,:) - stepsize.* Xuupdatei;
            X(n+j,:) = X(n+j,:) - stepsize.*Xdupdatej;
        end
        %}
        
        
        % method 2: matrix exponential
        C = symGXJ(1:n,n+1:end); % n by n matrix
        expC1 = expm(stepsize.*C); % n^3
        expC2 = expm(-stepsize.*C'); % n^3
        X = [expC1*X(1:n,:); expC2*X(n+1:end,:)]; % 8 n^2 p
        %X = [expC1*X(1:n,:); expC1' \ X(n+1:end,:)];

        % total flops: 2 n^3   +  32 n^2p  +  8 np + 4n^2


        timeperit = toc(t0);
        infos.time(it, 1) = timeperit;
        infos.cost(it, 1) = problem.cost(X);
        infos.gradnorm(it, 1) = problem.gradnorm(X);
        infos.optgap(it, 1) = problem.optgap(X);

        fprintf('%s:  %3d\t%+.3e \t%.3e\n', 'CDblock', it, infos.optgap(it), stepsize);

        if (sum(infos.time(1:it, 1)) > options.maxtime ) || infos.optgap(it, 1) < options.minoptgaptol
            break;
        end
            

    end

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];
    
    blockflops = 2 * (n^3)   +   32 * (n^2 )* p   +   8 *n *p + 4*n^2;
    infos.flops = [0; blockflops*ones(it,1)]; 

    infos.time = cumsum(infos.time);

end

