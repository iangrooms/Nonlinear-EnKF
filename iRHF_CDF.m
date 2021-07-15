function [out_prior,out_post,breaks] = iRHF_CDF(YN,ell)
% Output:
%   out_prior is the cdf of the prior evaluated at YN
%   out_post is a function handle that evaluates the cdf of the posterior
% Input:
%   YN is the prior ensemble
%   ell is a function handle that evaluates the likelihood l(x)

    Ne = length(YN);
    YN = sort(YN(:));
    mu = mean(YN);
    sigma = std(YN);
    
    % Get kernel widths
    h0 = 3.13*min(std(YN),iqr(YN)/1.34)/Ne^0.2;
    dY0 = 0.5*[0;diff(YN)];
    dY1 = 0.5*[dY0(2:end);0];
    h = max([h0*ones(Ne,1) dY0 dY1],[],2);
    
    % Get breaks and coefs for prior, ignoring tails
    [breaks,I] = sort([YN-h/2;YN+h/2]); 
    h_tmp = [1./h;-1./h];
    coefs = (1/Ne)*cumsum(h_tmp(I(1:end-1)));
    % = (1/Ne)*sum((abs(.5*(breaks(1:end-1)+breaks(2:end))' - YN) < h/2)./h)';
    
    % Use prior without tails to get out_prior = Phi_f(YN)
        % Set up the piecewise-constant prior pdf for the interior
        pp_prior = mkpp(breaks,coefs);
        % Integrate to get piecewise-linear prior cdf
        pp_prior = ppint(pp_prior);
        % Evaluate at YN
        out_prior = ppval(pp_prior,YN);

    % Set up function handle for posterior cdf
        % get piecewise-polynomial approximation of the likelihood
        LN = ell(breaks);
        pp_ell = pchip(breaks,LN);
        [~,coefs_ell,~,~,~] = unmkpp(pp_ell);
        % get piecewise-polynomial approximation of the un-normalized
        % posterior pdf without tails
        coefs = coefs.*coefs_ell; 
        pp_post = mkpp(breaks,coefs);
        % Integrate to get un-normalized piecewise-polynomial posterior cdf
        % without counting tails
        pp_cdf = ppint(pp_post);
        % Add the left tail correction to the interior cdf:
        [~,coefs,~,~,~] = unmkpp(pp_cdf); % get uncorrected coefficients
        coefs(:,end) = coefs(:,end) + LN(1)*normcdf(breaks(1),mu,sigma); % correct the coefficients
        pp_cdf = mkpp(breaks,coefs); % This is now the un-normalized interior part
        % The next bit gets the normalization constant
        Z = ppval(pp_cdf,breaks(end)) + LN(end)*normcdf(breaks(end),mu,sigma,'upper');
        % Finally put it all together
        out_post = @(y) post_cdf(y,mu,sigma,pp_cdf,[breaks(1) breaks(end)],Z,LN);
end

function post_cdf = post_cdf(y,mu,sigma,pp,endpoints,Z,LN)
% This function evaluates the iRHF's approximation  to the posterior CDF. 
% It puts together the pieces computed above.
    post_cdf = y;
    ind_cent = (y > endpoints(1)) & (y < endpoints(2));
    post_cdf(ind_cent) = ppval(pp,y(ind_cent));
    ind_left = (y <= endpoints(1));
    post_cdf(ind_left) = LN(1)*normcdf(y(ind_left),mu,sigma);
    ind_right = (y >= endpoints(2));
    post_cdf(ind_right) = ppval(pp,endpoints(2)) + LN(end)*(normcdf(endpoints(2),mu,sigma,'upper')-normcdf(y(ind_right),mu,sigma,'upper'));
    post_cdf = post_cdf/Z;
end
