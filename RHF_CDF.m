function out = RHF_CDF(YN,LN)
% The output is a handle to a function that evaluates the RHF's approximation
% to the posterior CDF
% YN is the prior ensemble
% LN is the likelihood evaluated at YN
% YN is allowed to take any real value, i.e. unbounded
% The likelihood is approximated as constant on the first and last interval

    YN = YN(:);
    LN = LN(:);

    % get parameters for the Gaussian tails
    mu = mean(YN);
    sigma = std(YN);

    Ne = length(YN);

    % Set up the piecewise-polynomial posterior pdf not counting tails
    [YN,I] = sort(YN);
    LN = LN(I);
    breaks = YN;
    coefs = (1/(Ne+1))*(1./diff(YN)).*[diff(LN)./diff(YN) LN(1:Ne-1)];
    pp_pdf = mkpp(breaks,coefs);
    % Integrate to get piecewise-polynomial posterior cdf not counting tails
    pp_cdf = ppint(pp_pdf);
    % Now add the left tail correction to the interior cdf:
    [~,coefs,~,~,~] = unmkpp(pp_cdf); % get uncorrected coefficients
    coefs(:,end) = coefs(:,end) + LN(1)/(Ne+1); % correct the coefficients
    pp_cdf = mkpp(breaks,coefs); % This is now the un-normalized interior part
    % The next bit gets the normalization constant
    Z = ppval(pp_cdf,YN(end)) + LN(end)/(Ne+1);
    out = @(y) post_cdf(y,mu,sigma,pp_cdf,[YN(1) YN(Ne)],Z,LN,Ne);
end

function post_cdf = post_cdf(y,mu,sigma,pp,endpoints,Z,LN,N)
% The output is a handle to a function that evaluates the RHF's approximation
% to the posterior CDF. It puts together the pieces computed above.
    post_cdf = y;
    ind_cent = (y > endpoints(1)) & (y < endpoints(2));
    post_cdf(ind_cent) = ppval(pp,y(ind_cent));
    ind_left = (y <= endpoints(1));
    post_cdf(ind_left) = LN(1)*normcdf(y(ind_left),mu,sigma)/((N+1)*(normcdf(endpoints(1),mu,sigma)));
    ind_right = (y >= endpoints(2));
    post_cdf(ind_right) = ppval(pp,endpoints(2)) + ...
             LN(end)*(normcdf(endpoints(2),mu,sigma,'upper')-normcdf(y(ind_right),mu,sigma,'upper'))/((N+1)*normcdf(endpoints(2),mu,sigma,'upper'));
    post_cdf = post_cdf/Z;
end
