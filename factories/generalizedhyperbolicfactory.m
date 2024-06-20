function M = generalizedhyperbolicfactory(n, p)
% Returns a manifold structure to optimize over generalized hyperbolic
% 
% function M = generalizedhyperbolicfactory(n, p)

    J = ones(n,1);
    J(1) = -1;

    inner = @(U, V) U(:).'*V(:);
    sym = @(A) (A + A')/2;
    
    if ~exist('p', 'var') || isempty(p)
        p = 1;
    end 

    M.name = @() sprintf('Generalized hyperbolic manifold GH(%d, %d)', n, p);

    M.dim = @() n*p - p*(p+1)/2;

    % generalized Lorentz inner product
    M.inner = @(X, U, V) generalized_lorentz_inner(U,V);
    function q = generalized_lorentz_inner(U, V)
        Vtemp = V; Vtemp(1,:) = -Vtemp(1,:);
        q = inner(U, Vtemp);
    end

    M.norm = @(x, d) sqrt(M.inner(X, d, d));

    M.proj = @projection;
    function Up = projection(X, U)
        Utemp = U;  Utemp(1,:) = -Utemp(1,:);
        Up = U + X*sym(X'*Utemp);
    end

    M.tangent = M.proj;
    
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        egtemp = egrad; egtemp(1,:) = -egtemp(1,:);
        rgrad = projection(X, egtemp);
    end
    M.ehess2rhess = @() error('generalizedhyperbolic.ehess2rhess not implemented yet.');
    
    M.retr = @retraction_cay;

    M.retr_cay = @retraction_cay;
    function Y = retraction_cay(X, U, t)
        JX = X; JX(1,:) = -JX(1,:);
        Px = eye(n) + 0.5.*(JX*X');
        Wu = 2.*skew(X*U'*Px);
        rhs = X + (t/2).*(Wu*JX);
        WuJ = Wu; WuJ(:,1) = -WuJ(:,1);
        lhs = eye(n) - (t/2).*(WuJ);
        Y = lhs \ rhs;
    end
    
    M.retr_exp = @retraction_exp;
    function Y = retraction_exp(X, U, t)
        JX = X; JX(1,:) = -JX(1,:);
        Px = eye(n) + 0.5.*(JX*X');
        Wu = 2.*skew(X*U'*Px);
        WuJ = Wu; WuJ(:,1) = -WuJ(:,1);
        Y = expm(t.*WuJ)*X;
    end
    
    M.rand = @() qr_unique(randn(n, p, k, array_type));
    

end

