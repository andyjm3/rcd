function M = symplecticfactory(n, p)
% Returns a manifold structure to optimize over symplectic Stiefel manifold.
%
% function M = symplecticfactory(n, p)   
% 
% The Symplectic manifold is the set of X \in R^2nx2p : X'*J_{2n}*X = J_{2p}
% The metric is such that the manifold is a Riemannian submanifold of R^2nx2p 
% equipped with the usual trace inner product, that is, it is the Euclidean metric.
% Another metric is canonical metric.
%
%
% The default retraction is quasi-geodesic-based retraction. The factory
% also implements other retractions, including Cayley transform and SR
% decomposition.
% 
% Reference:
%   Bin Gao, Nguyen Thanh Son, P.-A. Absil, Tatjana Stykel
%   1. Riemannian optimization on the symplectic Stiefel manifold
%   2. Geometry of the symplectic Stiefel manifold endowed with the Euclidean metric
%
% This factory file is based on the source code:
% https://github.com/opt-gaobin/spopt/blob/master/spopt.m

    skew = @(A) 0.5.*(A - A');
    sym = @(A) 0.5.*(A + A');
    
    if nargin < 2
        p = n;
    end

    assert(n >= p, 'The dimension n must be larger than the dimension p.');
    
    M.name = @() sprintf('Symplectic manifold Sp(%d, %d)', n, p);

    M.dim = @() 4*n*p - p*(2*p-1);

    M.inner = @(x, d1, d2) d1(:).'*d2(:);

    M.norm = @(x, d) norm(d(:));

    M.dist = @(x, y) error('symplectic.dist not implemented yet.');

    M.proj = @projection;
    function Up = projection(X, U)
        % total: 40 np^2 + 8p^3 + 8p^2 + 4np
        JX = [X(n+1:end,:); -X(1:n,:)];
        JXU = JX'*U; % 16 np^2
        skewJXU = JXU-JXU'; % 4p^2
        XX = X'*X; % 8np^2
        Omega = lyap(XX,-skewJXU); % 8p^3
        Omega = skew(Omega); % to ensure skew-symmetric (4p^2)
        Up = U - JX*Omega; % 16np^2+ 4np
    end

    M.tangent = M.proj;
    M.egrad2rgrad = M.proj;
    M.ehess2rhess = @() error('symplectic.ehess2rhess not implemented yet.');
    
    M.retr = @retraction_cl;

    M.retr_qg = @retraction_qg; % quasi-geodesic
    function Y = retraction_qg(X, U, t)
        % the source code is problematic. I follow the expression in eq (4.16) where Z = U
        % W = X'JU = - (JX)'U
        JX = [X(n+1:end,:); -X(1:n,:)]; % 2np
        W = - JX'*U; % 2 (8 n p^2)
        W = sym(W); % 6 p^2
        JW = [W(p+1:end,:); -W(1:p,:)]; % 2 p^2
        XU = [X, U]; 
        JZJZ = [U(:,p+1:end) -U(:,1:p)]'*[U(n+1:end,:); -U(1:n,:)]; % 16 np^2 + 4np
        H = [-JW JZJZ; eye(2*p) -JW]; % size is 4p by 4p
        ExpH = expm(t*H); % 64 p^3 + 16p^2
        Y =  XU*(ExpH(:,1:2*p)*expm(t*JW)); % 32 n p^2 + 32 p^3 + 4p^2 + 8 p^3
        % total: 64 np^2 + 104 p^3 +  28p^2 + 6 np
    end
    
    M.retr_cl = @retraction_cl;
    function Y = retraction_cl(X,U,t) % 32np^2 + 12 np + 16 np^2 + 16 p^2 + 8 p^3 + 16 np^2 
        % total: 64 np^2 + 8 p^3 + 16 p^2 + 12np
        XJ = [-X(:,p+1:end) X(:,1:p)]; % 2np
        JX = [X(n+1:end,:); -X(1:n,:)]; % 2np
        GX = eye(2*n) - 0.5*(XJ*JX'); % 
        SXZ = GX * U * XJ';
        SXZ = 2*sym(SXZ);
        SXZJ = [-SXZ(:,n+1:end), SXZ(:,1:n)];
        SXZJX = SXZ * JX;
        Y = (eye(2*n) - (0.5*t)*SXZJ) \ (X + (0.5*t)* SXZJX); 
    end

    M.retr_sr = @retraction_sr;
    function Y = retraction_sr(X, U, t)
        Y = MsGS(X + t*U);
    end


    M.rand = @randmfd;
    function Y = randmfd()
        W = randn(2*p,2*p); W = W'*W+0.1*eye(2*p); E = expm([W(p+1:end,:); -W(1:p,:)]);
        Y = [E(1:p,:);zeros(n-p,2*p);E(p+1:end,:);zeros(n-p,2*p)];
    end

    M.randvec = @randomvec;
    function U = randomvec(X)
        U = projection(X, randn(2*n, 2*p));
        U = U / norm(U(:));
    end
    
    M.lincomb = @matrixlincomb;
    M.transp = @(x1, x2, d) projection(x2, d);

    M.is_on_mfd = @is_on_mfd;
    function flag = is_on_mfd(X)
        Jp = [zeros(p,p) eye(p); -eye(p) zeros(p,p)];
        JX = [X(n+1:end,:); -X(1:n,:)];
        flag = ( norm(X'*JX - Jp, 'fro') < 1e-10);
    end
    

end

