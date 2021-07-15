%% Implements EnKF for L96.
% 2-step: RHF then LR
Driver_Setup
locRad = 11;
rInf = 1.00;
Ne = 20;
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
        ZN0 = ZNA;
        ell = @(t) normpdf(t,Y(k,ii),obsErr);
        [c_targets,post_cdf,breaks] = iRHF_CDF(ZN,ell);
        [~,I] = sort(ZN);
        % For values between the boundaries, set initial guess by interpolating.
          % First find the boundaries
          [pcdf_breaks,ind_unique,~] = unique(post_cdf(breaks),'stable'); % Eliminate duplicates
          left_ind = find(c_targets>min(pcdf_breaks),1,'first');
          right_ind = find(c_targets<max(pcdf_breaks),1,'last');
          % Now interpolate between the boundaries
          ZN0(left_ind:right_ind) = interp1(pcdf_breaks,breaks(ind_unique),...
              c_targets(left_ind:right_ind));
          % Initial guess in the tails is the edge of the tail
          ZN0(1:left_ind-1) = ZN(I(1));
          ZN0(right_ind+1:end) = ZN(I(Ne));
        % For any targets in the tails, use rootfinding
        for kk = 1:Ne
            ZNA(kk) = fzero(@(t) post_cdf(t) - c_targets(kk),ZN0(kk),opts);
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
FRMSE = squeeze(sqrt(mean((XT-FM).^2)));
ARMSE = squeeze(sqrt(mean((XT-AM).^2)));
save(sprintf('dt05/Ne%03d/L%1d/r%02d/DATA_iRHFLR_QUAD.mat',Ne,floor(locRad),0),...
    'FRMSE','ARMSE','FS','AS','FCRPS','ACRPS')
clear all