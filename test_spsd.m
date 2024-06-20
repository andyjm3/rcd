function test_spsd()
    % loss f(A) = || P.*A - PA  ||_F^2
    clear 
    clc
    rng(42);
    
    symm = @(Z) 0.5*(Z+Z');
    
    % parameters
    example =4;
    switch example
        case 1
            n = 100;
            p = 100;
            CN = 1000;
            data_choice = 'full'; % 'full' or 'sparse'

        case 2
            n = 500;
            p = 500;
            CN = 1000;
            data_choice = 'full';

        case 3
            n = 500;
            p = 300;
            CN = 1000;
            data_choice = 'full';

        case 4
            n = 500;
            p = 100;
            CN = 1000;
            data_choice = 'sparse';

        case 5
            n = 500;
            p = 500;
            CN = 1000;
            data_choice = 'sparse';
    end
        
    % generate true A
    D = 10*diag(logspace(-log10(CN), 0, p)); fprintf('Exponential decay of singular values with CN %d.\n \n\n', CN);
    D = diag([zeros(n-p,1); diag(D)]);
    [Q, R] = qr(randn(n)); %#ok
    A = Q*D*Q';
    
    % generate P
    switch data_choice
        case 'sparse'            
            fraction = 0.7;
            P = rand(n,n);
            P = 0.5*(P+P');
            P = (P <= fraction);            
        case 'full'
            P = ones(n);
    end
    
    % Hence, we know the nonzero entries in PA:
    PA = P.*A;
    
    % for AI and BW
    function f = cost(X)
        f = 0.5*norm(P.*X - PA, 'fro')^2;
    end
    
    function g = egrad(X)
        g = (P.*X - PA).*P;
    end    

    function mydist = optgap(X)
        mydist = norm(X - A, 'fro');
    end    

    function gn = gradnorm(X)
        U = egrad(X);
        gn = sqrt(sum(U(:).*U(:)));
    end

    function f = costYY(Y)
        f = cost(Y*Y');
    end

    function g = egradYY(Y)
        g = 2.*symm(egrad(Y*Y'))*Y;
    end
   
    function mygap = optgapYY(Y) 
        mygap = optgap(Y*Y');
    end

    function gn = gradnormYY(Y)
        U = egradYY(Y);
        gn = sqrt(sum(U(:).*U(:)));
    end

    % YY
    % problemYY.M = symfixedrankYYfactory(n,p);
    problemYY.cost = @costYY;
    problemYY.egrad = @egradYY;
    problemYY.optgap = @optgapYY;
    problemYY.gradnorm = @gradnormYY;

    %Y0 = randn(n,p);
    Y0 = eye(n,p);
    X0 = Y0*Y0';
    
    params = set_params(example);
   
    
    % YY
    options.stepsize = params.yy_lr;
    options.maxiter = 1000;
    options.method = 'YY';
    [Y, infos_yy] = RGD_spsd(problemYY, Y0, options);
    %}
    
    
    
    options.stepsize = params.cd_lr;
    options.update_type = 'noreplace';
    options.numupdates = params.cd_numupdates;
    options.maxiter = 1000;
    [Y, infos_cd] = coordinate_descent_spsd(problemYY,Y0, options);
    %}

    
    %flops
    egrad_flops = n^2;
    egrad_flops_yy = 2*n^2*p + 3*n^2 + 2*n^2*p;
    infos_yy.flops_yy(2:end,1) = egrad_flops_yy + infos_yy.flops_yy(2:end,1);
    infos_cd.flops(2:end,1) =  egrad_flops_yy + infos_cd.flops(2:end,1);


    
    %%

    lw = 1.3;
    ms = 2.3;
    axis_fs = 15;
    lg_fs = 18;

    colors = colororder(); % default colors
    

    h1 = figure(1);
    semilogy(1:length(infos_yy.cost), infos_yy.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy(1:length(infos_cd.cost), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'RGD', 'RCDlin'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Iteration','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);

    

    h2 = figure(2);
    semilogy(infos_yy.time, infos_yy.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy(infos_cd.time, infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'RGD', 'RCDlin'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Time','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);


    h3 = figure(3);
    semilogy(cumsum(infos_yy.flops_yy), infos_yy.optgap, '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy(cumsum(infos_cd.flops), infos_cd.optgap, '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', 'red'); hold on;
    hold off;
    ax = gca;
    set(gca, 'FontName', 'Arial');
    lg = legend({'RGD', 'RCDlin'}, 'NumColumns',1);
    lg.FontSize = lg_fs;
    ax.XAxis.FontSize = axis_fs;
    ax.YAxis.FontSize = axis_fs;
    xlabel(ax,'Flops','FontSize',23);
    ylabel(ax,'Optimality gap','FontSize',23);


    function params = set_params(example)
        switch example
            case 1
                params.ai_lr = 0.02;
                params.bw_lr = 0.04;
                params.yy_lr = 0.04;
                params.cd_lr = 0.2; 
                params.cd_numupdates = 2000;

            case 2
                params.ai_lr = 0.01;
                params.bw_lr = 0.04;
                params.yy_lr = 0.04;
                params.cd_lr = 0.2; 
                params.cd_numupdates = 50000;

            case 3
                params.yy_lr = 0.04;
                params.cd_lr = 0.24; 
                params.cd_numupdates = 30000;

            case 4
                params.yy_lr = 0.06;
                params.cd_lr = 0.3; 
                params.cd_numupdates = 10000;
            
            case 5
                params.yy_lr = 0.05;
                params.cd_lr = 0.2; 
                params.cd_numupdates = 50000;
                

        end
    end
    
end

