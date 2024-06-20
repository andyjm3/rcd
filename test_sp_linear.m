function test_sp_linear()

    % need to retune

    clear;
    clc;
    rng(42);

    example = 2;
    switch example
        case 1
            % large full rank 
            n = 400; 
            p = 400;
            maxiter = 1000;
        case 2
            % small full rank
            n = 200; 
            p = 200;
            maxiter = 200;
            minoptgaptol = 1e-5;
        case 3
            % low rank
            n = 200;
            p = 50;
            maxiter = 1000;
    end

    
    A = randn(2*n,2*p); 
    A = A/norm(A,2); 

    % problem 1: F-norm min
    function f = cost(X)
        f = norm(X - A, 'fro')^2/2;
    end

    function g = egrad(X) % 
        g = (X - A);
    end

    function g = egradrowi(X, i)
        g = X(i,:) - A(i,:);
    end

    egrad_flops = 4*n*p;
    egradrowi_flops = 2*p;

    function gn = gradnorm(X)
        UU = egrad(X);
        UU = mfd.egrad2rgrad(X, UU);
        gn = mfd.norm(X, UU);
    end

    function mygap = optgap(X)
        mygap = abs(cost(X) - costsol)/abs(costsol);
    end 

    % init
    mfd = symplecticfactory(n,p);
    X0 = mfd.rand();
    X0 = zeros(2*n,2*p); X0(1:p,1:p) = eye(p); X0(n+1:n+p,p+1:end) = eye(p);


    problem.M = mfd;
    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.egradrowi = @egradrowi;
    problem.gradnorm = @gradnorm;
    problem.optgap = @optgap;
    

    switch example
        case 2
            costsol = 1.2170693472847545e+02;
        case 3
            costsol = 1.9827444795283064e+01;
    end

    params = set_params(example);

    
    % cayley
    options.maxiter = maxiter;
    options.stepsize = params.cl_lr;
    options.minoptgaptol = minoptgaptol;
    options.method = 'CL';
    [X, infos_cl] = RGD_sp(problem, X0, options);


    % quasi-geodesic
    options.maxiter = maxiter;
    options.stepsize = params.qg_lr;
    options.minoptgaptol = minoptgaptol;
    options.method = 'QG';
    [X, infos_qg] = RGD_sp(problem, X0, options); 

    % SR
    options.maxiter = maxiter;
    options.stepsize = params.sr_lr;
    options.minoptgaptol = minoptgaptol;
    options.method = 'SR';
    [X, infos_sr] = RGD_sp(problem, X0, options);
    
    
    
    % CD-block
    options_cd_block.maxiter = maxiter;
    options_cd_block.stepsize = params.cdlinb_lr;
    options_cd_block.minoptgaptol = minoptgaptol;
    options_cd_block.linearization = 0; 
    [X, infos_cd_block] = coordinate_descent_symplectic_block(problem, X0, options_cd_block);
    
    % CDlin-block
    options_cd_block.maxiter = maxiter;
    options_cd_block.stepsize = params.cdb_lr;
    options_cd_block.minoptgaptol = minoptgaptol;
    options_cd_block.linearization = 1; 
    [X, infos_cdlin_block] = coordinate_descent_symplectic_block(problem, X0, options_cd_block);
    
    
    % CDlin
    options_cd.maxiter = maxiter;
    options_cd.stepsize = params.cdlin_lr;
    options_cd.minoptgaptol = minoptgaptol;
    options_cd.linearization = 1; 
    [X, infos_cdlin] = coordinate_descent_symplectic(problem, X0, options_cd);
%}
    
    % CD
    options_cd.maxiter = maxiter;
    options_cd.stepsize = params.cd_lr;
    options_cd.minoptgaptol = minoptgaptol;
    options_cd.linearization = 0; 
    [X, infos_cd] = coordinate_descent_symplectic(problem, X0, options_cd);

    infos_cl.flops_cl(2:end,1) = egrad_flops + infos_cl.flops_cl(2:end,1);
    infos_qg.flops_qg(2:end,1) = egrad_flops + infos_qg.flops_qg(2:end,1);
    infos_sr.flops_sr(2:end,1) = egrad_flops + infos_sr.flops_sr(2:end,1);
    infos_cdlin_block.flops(2:end,1) = egrad_flops + infos_cdlin_block.flops(2:end,1);
    infos_cd_block.flops(2:end,1) = 3*egrad_flops + infos_cd_block.flops(2:end,1);

    infos_cdlin.flops(2:end,1) = egrad_flops + infos_cdlin.flops(2:end,1);
    infos_cd.flops(2:end,1) = egradrowi_flops * infos_cd.inneriter + infos_cd.flops(2:end,1);
    
    

    %% Plots

    lw = 1.3;
    ms = 2.3;
    axis_fs = 15;
    lg_fs = 18;

    colors = colororder(); % default colors
    

    h1 = figure(1);
    semilogy(1:length(infos_qg.cost), infos_qg.optgap, '-d', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy(1:length(infos_cl.cost), infos_cl.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy(1:length(infos_sr.cost), infos_sr.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy(1:length(infos_cd.cost), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy(1:length(infos_cdlin.cost), infos_cdlin.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(2,:)); hold on;
    semilogy(1:length(infos_cd_block.cost), infos_cd_block.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    semilogy(1:length(infos_cdlin_block.cost), infos_cdlin_block.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'QG','CL','SR','RCD', 'RCDlin', 'RCD-block', 'RCDlin-block'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Iteration','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);

    

    h2 = figure(2);
    semilogy(infos_qg.time, infos_qg.optgap, '-d', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy(infos_cl.time, infos_cl.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy(infos_sr.time, infos_sr.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy(infos_cd.time, infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy(infos_cdlin.time, infos_cdlin.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(2,:)); hold on;
    semilogy(infos_cd_block.time, infos_cd_block.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    semilogy(infos_cdlin_block.time, infos_cdlin_block.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'QG','CL','SR','RCD', 'RCDlin', 'RCD-block', 'RCDlin-block'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Time','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);


    h3 = figure(3);
    semilogy(cumsum(infos_qg.flops_qg), infos_qg.optgap, '-d', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy(cumsum(infos_cl.flops_cl), infos_cl.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy(cumsum(infos_sr.flops_sr), infos_sr.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy(cumsum(infos_cd.flops), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy(cumsum(infos_cdlin.flops), infos_cdlin.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', colors(2,:)); hold on;
    semilogy(cumsum(infos_cd_block.flops), infos_cd_block.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    semilogy(cumsum(infos_cdlin_block.flops), infos_cdlin_block.optgap, '-.', 'MarkerSize',ms, 'LineWidth',lw+0.2, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'QG','CL','SR','RCD', 'RCDlin', 'RCD-block', 'RCDlin-block'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Flops','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);
    

    %%
    function params = set_params(example)
        switch example
            case 2
                params.qg_lr = 1; 
                params.cl_lr = 1;
                params.sr_lr = 1;
                params.cdlinb_lr = 0.4;
                params.cdb_lr = 0.4;
                params.cdlin_lr = 0.4;
                params.cd_lr = 0.4;

            case 3
                params.qg_lr = 1; 
                params.cl_lr = 1;
                params.sr_lr = 1;
                params.cdlinb_lr = 0.65; 

        end
    end

end

