function test_pro_infea()
    % test RCD aginst infeasible methods
    clear; clc;
    rng(42); % rng(20)

    % profile on
    sym = @(P) (P + P')/2;
    skew = @(P) (P - P')/2;
    inner = @(U, V) U(:)'*V(:);

    example = 3;
    
    switch example
        case 1
            n = 50;
            p = 50;
            maxiter = 1000;
            minoptgaptol = 1e-5;
        case 2
            n = 200;
            p = 200;
            maxiter = 1500;
            minoptgaptol = 1e-5;
        case 3
            n = 200;
            p = 150;
            maxiter = 1000;
            minoptgaptol = 1e-5;
        case 4
            n = 200;
            p = 50;
            maxiter = 1000;
            minoptgaptol = 1e-5;
    end

    mfd = stiefelfactorynew(n,p);
    
    A = randn(p,p);
    B = randn(n,p);

    BAt = B * A';
    AAt = A*A';
    
    [U, ~, V] = svd(B*A', 0);
    Xsol = U*V';
    costsol = cost(Xsol);
    
    X0 = randn(n,p);
    [X0, ~] = qr(X0,0);


    function f = cost(X)
        f = -trace(X'*BAt)/p;
    end

    function g = egrad(X) % flops: 0
        g = - BAt/p;
    end

    function gn = gradnorm(X)
        UU = egrad(X);
        UU = mfd.egrad2rgrad(X, UU);
        gn = mfd.norm(X, UU);
    end

    function mygap = optgap(X) % For stats and plotting
        
        mygap = abs(cost(X) - costsol)/abs(costsol);

    end


    % init for ExpPen and other infeasible methods
    function f = cost_infea(X)
        f = norm(X * A - B,'fro')^2/(2*p);
    end

    function g = egrad_infea(X) % 2np^2 + 2np
        % egrad for infeasible methods like ExPen, PLAM, PCAL, PenCf, and Landing
        g = (X* AAt - BAt)/p;
    end

    egrad_flops = 0;
    egrad_infea_flops = 2*n*(p^2) + 2*n*p;

    problem.M = stiefelfactorynew(n,p);
    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.gradnorm = @gradnorm;
    
    problem.cost_infea = @cost_infea;
    problem.egrad_infea = @egrad_infea;
    
    problem.optgap = @optgap;
    
    params = set_params(example);


    % Pencf
    options_pencf.maxiter = maxiter;
    options_pencf.stepsize = params.pencf_lr;
    options_pencf.stepsize_type = 'BB';
    options_pencf.lam = params.pencf_lam;
    options_pencf.minoptgaptol = minoptgaptol;

    [X_pencf, infos_pencf] = pencf(problem, X0, options_pencf);
    

    
    % Pcal
    options_pcal.maxiter = maxiter;
    options_pcal.stepsize = params.pcal_lr;
    options_pcal.stepsize_type = 'BB';
    options_pcal.lam = params.pcal_lam;
    options_pcal.minoptgaptol = minoptgaptol;

    [X_pcal, infos_pcal] = pcal(problem, X0, options_pcal);    

    
    
    % Plam
    options_plam.maxiter = maxiter;
    options_plam.stepsize = params.plam_lr;
    options_plam.stepsize_type = 'BB';
    options_plam.lam = params.plam_lam;
    options_plam.minoptgaptol = minoptgaptol;

    [X_plam, infos_plam] = plam(problem, X0, options_plam);
    
    
   
    
    % Landing
    options_landing.maxiter = maxiter;
    options_landing.stepsize = params.ld_lr;
    options_landing.safe_stepsize = params.safe_stepsize;
    options_landing.lam = params.ld_lam;
    options_landing.eps = params.ld_eps;
    options_landing.minoptgaptol = minoptgaptol;

    [X_landing, infos_landing] = landing(problem, X0, options_landing);
    
        

    % ExPen
    options_expen.lam = params.expen_lam;
    options_expen.maxiter = maxiter;
    options_expen.stepsize = params.expen_lr; % 0.3; for CG
    options_expen.method = 'SD';
    options_expen.minoptgaptol = minoptgaptol;

    [X_expen, infos_expen] = ExPen(problem, X0, options_expen);
    
    % RCD
    nn =  n*(n-1)/2; 
    options_cd.maxiter = maxiter;
    options_cd.stepsize = params.cd_lr;
    options_cd.numupdates =  min(nn, floor(nn)); % n*p - p*(p+1)/2; 10p*numupdates, 10np^2 - 5np^2 > 5np^2
    options_cd.update_type = 'cyclic';
    options_cd.minoptgaptol = minoptgaptol;

    [X_cd, infos_cd] = coordinate_descent_orthogonal(problem, X0, options_cd);


    infos_cd.flops(2:end,1) = egrad_flops + infos_cd.flops(2:end,1);
    infos_expen.flops_sd(2:end,1) = egrad_infea_flops +  infos_expen.flops_sd(2:end,1);
    infos_landing.flops_fix(2:end,1) = egrad_infea_flops + infos_landing.flops_fix(2:end,1);
    infos_pcal.flops_BB(2:end,1) = egrad_infea_flops + infos_pcal.flops_BB(2:end,1);
    infos_plam.flops_BB(2:end,1) = egrad_infea_flops + infos_plam.flops_BB(2:end,1);
    infos_pencf.flops_BB(2:end,1) = egrad_infea_flops + infos_pencf.flops_BB(2:end,1);
    

    %%
    lw = 1.3;
    ms = 2.3;
    axis_fs = 15;
    lg_fs = 18;

    colors = colororder(); % default colors

    h1 = figure(1);    
    semilogy(1:length(infos_expen.cost), infos_expen.optgap, '-d', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(7,:)); hold on;
    semilogy(1:length(infos_landing.cost), infos_landing.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(6,:)); hold on;
    semilogy(1:length(infos_plam.cost), infos_plam.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(5,:)); hold on;
    semilogy(1:length(infos_pcal.cost), infos_pcal.optgap, '-s', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy(1:length(infos_pencf.cost), infos_pencf.optgap, '-x', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy(1:length(infos_cd.cost), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'ExPen','Landing','PLAM','PCAL','PenCF', 'RCD'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    ylim([10e-7, 1]);
    xlabel(ax,'Iteration','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);



    h2 = figure(2);
    semilogy(infos_expen.time, infos_expen.optgap, '-d', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(7,:)); hold on;
    semilogy(infos_landing.time, infos_landing.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(6,:)); hold on;
    semilogy(infos_plam.time, infos_plam.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(5,:)); hold on;
    semilogy(infos_pcal.time, infos_pcal.optgap, '-s', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy(infos_pencf.time, infos_pencf.optgap, '-x', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy(infos_cd.time, infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'ExPen','Landing','PLAM','PCAL','PenCF', 'RCD'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    ylim([10e-7, 1]);
    xlabel(ax,'Time','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);



    h3 = figure(3);
    semilogy(cumsum(infos_expen.flops_sd), infos_expen.optgap, '-d', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(7,:)); hold on;
    semilogy(cumsum(infos_landing.flops_fix), infos_landing.optgap, '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(6,:)); hold on;
    semilogy(cumsum(infos_plam.flops_BB), infos_plam.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(5,:)); hold on;
    semilogy(cumsum(infos_pcal.flops_BB), infos_pcal.optgap, '-s', 'MarkerSize',ms, 'LineWidth',lw,'color', colors(4,:)); hold on;
    semilogy(cumsum(infos_pencf.flops_BB), infos_pencf.optgap, '-x', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy(cumsum(infos_cd.flops), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'ExPen','Landing','PLAM','PCAL','PenCF', 'RCD'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    %legend boxoff
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    ylim([10e-7, 1]);
    xlabel(ax,'Flops','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);


    function params = set_params(example)
        switch example
            case 1
                % pencf
                params.pencf_lr = 1;   params.pencf_lam = 3;
                % pcal
                params.pcal_lr = 1;    params.pcal_lam = 3;
                % plam
                params.plam_lr = 1;    params.plam_lam = 3;
                % Landing
                params.ld_lam = 1;    params.ld_eps = 0.5; 
                params.ld_lr = 0.6;
                params.safe_stepsize = false;  
                % expen
                params.expen_lam = 3;   params.expen_lr = 0.15;
                % EXP
                params.exp_lr = 0.3;
                % QR
                params.qr_lr = 0.8; 
                % CL
                params.cl_lr = 0.8;
                % CD
                params.cd_lr = 1.1;   
                params.cdrand_lr = 1.0;
            case 2
                % pencf 
                params.pencf_lr = 1;    params.pencf_lam = 3;
                % pcal
                params.pcal_lr = 1;    params.pcal_lam = 3;
                % plam
                params.plam_lr = 1;    params.plam_lam = 3;
                % Landing
                params.ld_lam = 1;    params.ld_eps = 0.5; 
                params.ld_lr = 0.6;
                params.safe_stepsize = false; 
                % expen
                params.expen_lam = 3;   params.expen_lr = 0.2;
                % EXP
                params.exp_lr = 0.4;
                % QR
                params.qr_lr = 0.8; 
                % CL
                params.cl_lr = 0.8;
                % CD
                params.cd_lr = 1.2;  
                params.cdrand_lr = 1.1;
            case 3
                params.pencf_lr = 1;    params.pencf_lam = 4;
                params.pcal_lr = 1;    params.pcal_lam = 3;
                params.plam_lr = 1;    params.plam_lam = 3;
                % Landing
                params.ld_lam = 1;    params.ld_eps = 1; 
                params.ld_lr = 0.5;
                params.safe_stepsize = false; 
                % expen
                params.expen_lam = 3;   params.expen_lr = 0.2;
                % EXP
                params.exp_lr = 0.3;
                % QR
                params.qr_lr = 0.7; 
                % CL
                params.cl_lr = 0.7;
                % 
                params.cd_sti = 1;
                params.cd_stirand = 1;
                % CD
                params.cd_lr = 1.3;   
                params.cdrand_lr = 1.3;
            case 4
                params.pencf_lr = 1;    params.pencf_lam =2;
                params.pcal_lr = 1;    params.pcal_lam = 1;
                params.plam_lr = 1;    params.plam_lam = 2;
                % landing
                params.ld_lam = 1;    params.ld_eps =1; 
                params.ld_lr = 0.4;
                params.safe_stepsize = false;
                % expen
                params.expen_lam = 1;   params.expen_lr = 0.2;
                % QR, CL
                params.exp_lr = 0.2;
                params.qr_lr = 0.4; 
                params.cl_lr = 0.4;
                %
                params.cd_sti = 0.5;
                params.cd_stirand = 0.4;
                % CD
                params.cd_lr = 1.5; 
                params.cdrand_lr = 2.;
        end

    end
end

