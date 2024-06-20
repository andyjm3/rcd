function test_pca_random()
    clear; clc;
    rng(42);

    % profile on
    sym = @(P) (P + P')/2;
    skew = @(P) (P - P')/2;
    inner = @(U, V) U(:)'*V(:);
    
    example = 1;

    switch example
        case 1
            n = 200;
            p = 50;
            maxiter = 2000;
            minoptgaptol = 1e-5;
            maxitercd = 1000;
        case 2
            n = 200;
            p = 100;
            maxiter = 5000;
            minoptgaptol = 1e-5;
            maxitercd = 500;
    end

    mfd = grassmannfactorynew(n,p);

    CN = 1000;
    D = 1000*diag(logspace(-log10(CN), 0, n)); fprintf('Exponential decay of singular values with CN %d.\n \n\n', CN);
    [Q, R] = qr(randn(n)); %#ok
    A = Q*D*Q';
    A = (A + A')/2;

    
    X0 = randn(n,p);
    [X0, ~] = qr(X0,0);


    function f = cost(X)
        AX = A*X;
        f = -0.5*inner(X,AX);
        f = f/n;
    end

    function g = egrad(X) 
        g = - A*X;        
        g = g/n;
    end

    function g = egradrowi(X,i)
        g = - A(i,:)*X;
        g = g/n;
    end

    function g = egradcoli(X,i)
        g = - A * X(:,i);
        g = g/n;
    end
    

    function gn = gradnorm(X)
        U = egrad(X);
        U = U - X*(X'*U); 
        gn = mfd.norm(X, U);
    end

    function mygap = optgap(X) % For stats and plotting   
        mygap = mfd.dist(X, Xsol);
    end


    % init for ExpPen and other infeasible methods
    function f = cost_infea(X)
        f = cost(X);
    end

    function g = egrad_infea(X) % 2np^2 + 2np
        % egrad for infeasible methods like ExPen, PLAM, PCAL, PenCf, and Landing
        g = egrad(X);
    end

    egrad_flops = 2*(n^2)*p + n*p; % change this for RCD and TSD
    egradcoli_flops = 2*n^2 + n;
    egradrowi_flops = 2*n*p + p;
    egrad_infea_flops = egrad_flops;

    %%
    problem.M = mfd;
    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.egradrowi = @egradrowi;
    problem.egradcoli = @egradcoli;
    problem.gradnorm = @gradnorm;
    
    problem.cost_infea = @cost_infea;
    problem.egrad_infea = @egrad_infea;
    
    problem.optgap = @optgap;
    

    [Xsol, D] = eigs(A/(2*n),p);
    costsol = cost(Xsol);
    
    params = set_params(example);

    
    % TSD (cyclic)
    options_tsd.maxiter = maxitercd;
    options_tsd.stepsize = params.cd_sti;
    options_tsd.linearization = false;
    options_tsd.update_type = 'cyclic';
    options_tsd.numupdates = p*(p+1)/2;
    options_tsd.minoptgaptol = minoptgaptol;
    [X_tsd, infos_tsd] =coordinate_descent_stiefel(problem, X0, options_tsd);
    %}

    
    % TSD (random)
    options_tsdrand.maxiter = maxitercd;
    options_tsdrand.stepsize = params.cd_stirand;
    options_tsdrand.linearization = false;
    options_tsdrand.update_type = 'replace';
    options_tsdrand.numupdates = p*(p+1)/2;
    options_tsdrand.minoptgaptol = minoptgaptol;
    [X_tsdrand, infos_tsdrand] =coordinate_descent_stiefel(problem, X0, options_tsdrand);
    %}
    
    
    % RCD (cyclic)
    nn = n*n - n*(n+1)/2;
    options_cd.maxiter = maxitercd;
    options_cd.stepsize = params.cd_lr;
    options_cd.linearization =false;
    options_cd.update_type = 'cyclic';
    options_cd.numupdates = nn;
    options_cd.minoptgaptol = minoptgaptol;

    [X_cd, infos_cd] = coordinate_descent_orthogonal(problem, X0, options_cd);
    %}
    
    % RCD (random)
    nn = n*n - n*(n+1)/2;
    options_cdrand.maxiter = maxitercd;
    options_cdrand.stepsize = params.cdrand_lr;
    options_cdrand.linearization =false;
    options_cdrand.update_type = 'replace';
    options_cdrand.numupdates = nn;
    options_cdrand.minoptgaptol = minoptgaptol;

    [X_cdrand, infos_cdrand] = coordinate_descent_orthogonal(problem, X0, options_cdrand);
    %}
    
    
    % RCDlin (cyclic)
    nn = n*n - n*(n+1)/2;
    options_cdlin.maxiter = maxitercd;
    options_cdlin.linearization =true;
    options_cdlin.numupdates = nn;  
    options_cdlin.stepsize = params.cdlin_lr;
    options_cdlin.update_type = 'cyclic';
    options_cdlin.minoptgaptol = minoptgaptol;

    [X_cd, infos_cdlin] = coordinate_descent_orthogonal(problem, X0, options_cdlin);
    %}
    
    
    % RCDlin (random full)
    nn = n*n - n*(n+1)/2;
    options_cdlinrf.maxiter = maxitercd;
    options_cdlinrf.linearization =true;
    options_cdlinrf.numupdates = nn;  
    options_cdlinrf.stepsize = params.cdlinrf_lr;
    options_cdlinrf.update_type = 'replace';
    options_cdlinrf.minoptgaptol = minoptgaptol;

    [X_cd, infos_cdlinrf] = coordinate_descent_orthogonal(problem, X0, options_cdlinrf);
    %}

    % RCDlin (random low)
    nn = p*(n-p);
    options_cdlinrl.maxiter = maxitercd;
    options_cdlinrl.linearization =true;
    options_cdlinrl.numupdates = nn;  
    options_cdlinrl.stepsize = params.cdlinrl_lr;
    options_cdlinrl.update_type = 'noreplace';
    options_cdlinrl.minoptgaptol = minoptgaptol;

    [X_cd, infos_cdlinrl] = coordinate_descent_orthogonal(problem, X0, options_cdlinrl);

    if n > p
        infos_tsd.flops(2:end,1) = egradcoli_flops * infos_tsd.inneriter + infos_tsd.flops(2:end,1);
        infos_tsdrand.flops(2:end,1) = egradcoli_flops * infos_tsdrand.inneriter + infos_tsdrand.flops(2:end,1);
    end
    infos_cd.flops(2:end,1) = egradrowi_flops * infos_cd.inneriter + infos_cd.flops(2:end,1);
    infos_cdrand.flops(2:end,1) = egradrowi_flops * infos_cdrand.inneriter + infos_cdrand.flops(2:end,1);
    infos_cdlin.flops(2:end,1) = egrad_flops + infos_cdlin.flops(2:end,1);
    infos_cdlinrf.flops(2:end,1) = egrad_flops + infos_cdlinrf.flops(2:end,1);
    infos_cdlinrl.flops(2:end,1) = egrad_flops + infos_cdlinrl.flops(2:end,1);
    
    %% plot
    lw = 1.3;
    ms = 2.3;
    axis_fs = 15;
    lg_fs = 18;

    colors = colororder(); % default colors

    h1 = figure(1);    
    if n > p
        semilogy(1:length(infos_tsdrand.cost), infos_tsdrand.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw,'color',colors(1,:)); hold on;
        semilogy(1:length(infos_tsd.cost), infos_tsd.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'blue'); hold on; 
    end
    semilogy(1:length(infos_cdrand.cost), infos_cdrand.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy(1:length(infos_cd.cost), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    semilogy(1:length(infos_cdlinrf.cost), infos_cdlinrf.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(2,:)); hold on;
    semilogy(1:length(infos_cdlinrl.cost), infos_cdlinrl.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(3,:)); hold on;
    semilogy(1:length(infos_cdlin.cost), infos_cdlin.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    if n > p
        lg = legend({'TSD-r', 'TSD-c', 'RCD-r', 'RCD-c', 'RCDlin-r', 'RCDlin-nr', 'RCDlin-c'}, 'NumColumns',1); 
    else
        lg = legend({'RCD-c', 'RCD-r'}, 'NumColumns',1);  
    end
    lg.FontSize = lg_fs;
    %legend boxoff
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Iteration','FontSize',23);
    ylabel(ax,'Distance to solution','FontSize',23);

    h2 = figure(2);
    if n > p
        semilogy(infos_tsdrand.time, infos_tsdrand.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw,'color',colors(1,:)); hold on;
        semilogy(infos_tsd.time, infos_tsd.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'blue'); hold on; 
    end
    semilogy(infos_cdrand.time, infos_cdrand.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy(infos_cd.time, infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    semilogy(infos_cdlinrf.time, infos_cdlinrf.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(2,:)); hold on;
    semilogy(infos_cdlinrl.time, infos_cdlinrl.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(3,:)); hold on;
    semilogy(infos_cdlin.time, infos_cdlin.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    if n > p
        lg = legend({'TSD-r', 'TSD-c', 'RCD-r', 'RCD-c', 'RCDlin-r', 'RCDlin-nr', 'RCDlin-c'}, 'NumColumns',1); 
    else
        lg = legend({'RCD-c', 'RCD-r'}, 'NumColumns',1);  
    end
    lg.FontSize = lg_fs;
    %legend boxoff
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Time','FontSize',23);
    ylabel(ax,'Distance to solution','FontSize',23);



    h3 = figure(3);
    if n > p
        semilogy(cumsum(infos_tsdrand.flops), infos_tsdrand.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw,'color',colors(1,:)); hold on;
        semilogy(cumsum(infos_tsd.flops), infos_tsd.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'blue'); hold on; 
    end
    semilogy(cumsum(infos_cdrand.flops), infos_cdrand.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy(cumsum(infos_cd.flops), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    semilogy(cumsum(infos_cdlinrf.flops), infos_cdlinrf.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(2,:)); hold on;
    semilogy(cumsum(infos_cdlinrl.flops), infos_cdlinrl.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(3,:)); hold on;
    semilogy(cumsum(infos_cdlin.flops), infos_cdlin.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    if n > p
        lg = legend({'TSD-r', 'TSD-c', 'RCD-r', 'RCD-c', 'RCDlin-r', 'RCDlin-nr', 'RCDlin-c'}, 'NumColumns',1); 
    else
        lg = legend({'RCD-c', 'RCD-r'}, 'NumColumns',1);  
    end
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Flops','FontSize',23);
    ylabel(ax,'Distance to solution','FontSize',23);


    function params = set_params(example)
        switch example
            case 1
                % EXP
                params.exp_lr = 0.3;
                % QR
                params.qr_lr = 0.3; 
                % CL
                params.cl_lr = 0.3;
                % TSD
                params.cd_sti = 0.3;
                params.cd_stirand = 0.3;
                % CD
                params.cd_lr = 2;   
                params.cdrand_lr = 1.9;
                params.cdlin_lr = 1.5;
                params.cdlinrf_lr = 1.6;
                params.cdlinrl_lr = 1.4;
            case 2
                % EXP
                params.exp_lr = 0.4;
                % QR
                params.qr_lr = 0.4; 
                % CL
                params.cl_lr = 0.4;
                % TSD
                params.cd_sti = 0.8;
                % CD
                params.cd_lr = 2.8;   
                params.cdlin_lr = 1.3;
                
        end
    end
end

