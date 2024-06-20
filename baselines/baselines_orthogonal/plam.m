function [X, infos] = plam(problem, X0, options)
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
        
        egradst = problem.egrad_infea(X);
        GtX = egradst' * X; % 2np^2
        GtXsym = sym(GtX); % 2p^2
        XX = X'*X; % np^2
        penalFeaX = options.lam * (XX - eye(p)); % 2 p^2
        dir = egradst - X *GtXsym + X*penalFeaX; % np + 2 np^2 + p^2

        % stepsize (BB1)
        if strcmpi(options.stepsize_type, 'BB') % total: 6np
            if it == 1 
                stepsize = max(0.1,min(0.01*norm(dir,'fro'),1));
            else
                Sk = X-Xprev;
                Vk = dir-dirprev;    % Vk = G-Gk;
                SV = Sk(:)'*Vk(:); %sum(sum(Sk.*Vk));
                proxparam = norm(Vk, 'fro')^2/abs(SV); %sum(sum(Vk.*Vk))/abs(SV); % SBB
                stepsize = 1/max(0,min(proxparam,1000));
            end
            Xprev = X;
            dirprev = dir;
            %flopi = flopi + 7*n*p;
        elseif strcmpi(options.stepsize_type, 'Fix')
            % fixed stepsize
            stepsize = options.stepsize;
        end

        X = X - stepsize * dir; %  2 np

        timeperit = toc(t0);

        % total flops: BB: 5np^2 + 5p^2 + 9np
        % total flops: fix: 5np^2 + 5p^2 + 3np


        infos.time(it, 1) = timeperit;
        infos.cost(it, 1) = problem.cost(X);
        infos.gradnorm(it, 1) = problem.gradnorm(X);
        infos.optgap(it, 1) = problem.optgap(X);

        fprintf('%s:  %3d\t%+.3e \t%.3e\n', 'PLAM', it, infos.optgap(it), stepsize);

        if sum(infos.time(1:it, 1)) > options.maxtime || infos.optgap(it, 1) < options.minoptgaptol
            break;
        end
    end

    infos.time = [0; infos.time(1:it,1)];
    infos.cost = [problem.cost(X0); infos.cost(1:it,1)];
    infos.gradnorm = [problem.gradnorm(X0); infos.gradnorm(1:it,1)];
    infos.optgap = [problem.optgap(X0); infos.optgap(1:it,1)];
    
    BBflops = 5*n*(p^2) + 5*(p^2) + 9*n*p;
    fixflops = 5*n*(p^2) + 5*(p^2) + 3*n*p;

    if strcmpi(options.stepsize_type, 'BB')
        infos.flops_BB = [0; BBflops*ones(it,1)];
    elseif strcmpi(options.stepsize_type, 'Fix')
        infos.flops_fix = [0; fixflops*ones(it,1)];
    end

    infos.time = cumsum(infos.time);

    X = qr_unique(X);
end