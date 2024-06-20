function [X, infos] = pcal(problem, X0, options)
    [n p] = size(X0);

    sym = @(P) (P + P')/2;
    skew = @(P) (P - P')/2;

    localdefaults.maxiter = 100;
    localdefaults.stepsize = 1;
    localdefaults.lam = 1;
    localdefaults.stepsize_type = 'BB';
    localdefaults.maxtime = inf;
    localdefaults.minoptgaptol = 1e-3;


    options = mergeOptions(localdefaults, options);

    maxiter = options.maxiter;   
    
    infos.time = nan(maxiter,1);
    infos.cost = nan(maxiter, 1);
    infos.gradnorm = nan(maxiter,1);
    infos.optgap = nan(maxiter,1);


    X = X0;
   
    for it = 1:maxiter

        t0 = tic();
        
        egradst = problem.egrad_infea(X); % Problem dependent.
        GtX = egradst' * X; % 2 n p^2.
        GtXsym = sym(GtX); % 2 p^2.
        XX = X'*X; % np^2
        penalFeaX = options.lam * (XX - eye(p)); % 2 p^2
        d = diag( GtX' - XX*GtXsym + XX*penalFeaX); % 3p^2  
        dir = egradst - X *GtXsym - X.*d' + X*penalFeaX; % 2np^2 + p^2 + 3np


        % stepsize %
        if strcmpi(options.stepsize_type, 'BB') % total: 6np
            % stepsize (BB1)
            if it == 1 
                stepsize = max(0.1,min(0.01*norm(dir,'fro'),1));
            else
                Sk = X-Xprev;   % np 
                Vk = dir-dirprev;    % np 
                SV = Sk(:)'*Vk(:); %sum(sum(Sk.*Vk)); % 2 np
                proxparam = norm(Vk, 'fro')^2/abs(SV); % 2 np
                stepsize = 1/max(0,min(proxparam,1000));
            end
            Xprev = X;
            dirprev = dir;
        elseif strcmpi(options.stepsize_type, 'Fix')  
            % fixed stepsize
            stepsize = options.stepsize;
        end
       
        X = X - stepsize * dir; % 2 np
        nx = sqrt(sum(X.*X)); % 2 np 
        X = X ./ nx; % np

        % total flops: BB: 5np^2 + 8p^2  + 14np
        % total flops: fix: 5np^2 + 8p^2 + 8np

        
        timeperit = toc(t0);

        infos.time(it, 1) = timeperit;
        infos.cost(it, 1) = problem.cost(X);
        infos.gradnorm(it, 1) = problem.gradnorm(X);
        infos.optgap(it, 1) = problem.optgap(X);


        fprintf('%s:  %3d\t%+.3e \t%.3e\n', 'PCAL', it, infos.optgap(it), stepsize);

        if sum(infos.time(1:it, 1)) > options.maxtime || infos.optgap(it, 1) < options.minoptgaptol
            break;
        end
    end

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];


    BBflops = 5*n*(p^2) + 8*(p^2) + 14*n*p;
    fixflops = 5*n*(p^2) + 8*(p^2) + 8*n*p;

    
    if strcmpi(options.stepsize_type, 'BB')
        infos.flops_BB = [0; BBflops*ones(it,1)];
    elseif strcmpi(options.stepsize_type, 'Fix')
        infos.flops_fix = [0; fixflops*ones(it,1)];
    end


    infos.time = cumsum(infos.time);

    X = qr_unique(X);
end