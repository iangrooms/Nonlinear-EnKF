%% Implements EnKF for L96.
% 2-step: RHF then LR
setup

% Run filter
opts = optimset('TolX',1E-4);%,'Display','off');
for ii=1:Nt
    % Forecast mean and spread
    FM(:,ii) = mean(X,2);
    FS(ii) = sqrt(mean(var(X,0,2)));
    % Compute CRPS at each point
    for jj=1:40
        FCRPS(jj,ii) = getCRPS(X(jj,:),ones(Ne,1)/Ne,XT(jj,ii));
    end
    % Get obs err ensemble
    obs_err = obsErr*randn(40,Ne);
    obs_err = obs_err - mean(obs_err,2);
    % Prior inflation:
    X = FM(:,ii) + rInf*(X - FM(:,ii));
    % Loop over obs
    for k=1:40
        % Step 1: ZN
        % Via RHF, so (i) get ZN
        ZN = X(k,:);
        % Get increments (i.e. update ZN)
        ZNA = ZN;
        ell = @(t) normpdf(t,Y(k,ii),obsErr);
        [c_targets,post_cdf,breaks] = iRHF_CDF(ZN,ell);
        [~,I] = sort(ZN);
        % For values between the boundaries, invert by interpolating.
          % First find the boundaries
          [pcdf_breaks,ind_unique,~] = unique(post_cdf(breaks),'stable'); % Eliminate duplicates
          left_ind = min([1;find(c_targets>min(pcdf_breaks),1,'first')]);
          right_ind = max([Ne;find(c_targets<max(pcdf_breaks),1,'last')]);
          % Now interpolate between the boundaries
          ZNA(left_ind:right_ind) = interp1(pcdf_breaks,breaks(ind_unique),...
              c_targets(left_ind:right_ind));
          if(any(isnan(ZNA))), return,end
        % For any targets in the tails, use rootfinding
        for kk = I((I<left_ind) | (I>right_ind))
            ZNA(kk) = fzero(@(t) post_cdf(t) - c_targets(kk),ZN(I(kk)),opts);
        end
        ZNA(I) = ZNA;
        dZN = ZNA - ZN; % These are the increments
        % Step 2: Linear Regression
        beta = L(:,k).*((X - mean(X,2))*(ZN'-mean(ZN)))./sum((ZN-mean(ZN)).^2);
        X = X + beta*dZN;
    end
    % Store EnKF analysis mean
    AM(:,ii) = mean(X,2);
    % Store EnKF posterior spread
    AS(ii) = sqrt(mean(var(X,0,2)));
    % Compute CRPS at each point
    for jj=1:40
        ACRPS(jj,ii) = getCRPS(X(jj,:),ones(Ne,1)/Ne,XT(jj,ii));
    end
    if(any(abs(X(:))>50))
        break
    end
    % forecast ensemble
    parfor jj=1:Ne
        [~,sol] = ode45(@RHS,[0 dtObs/2 dtObs],X(:,jj));
        X(:,jj) = sol(3,:)';
    end
end
