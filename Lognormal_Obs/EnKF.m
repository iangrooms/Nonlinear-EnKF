%% Implements EnKF for L96.
setup

% Run filter
for ii=1:Nt
    % Forecast mean and spread
    FM(:,ii) = mean(X,2);
    FS(ii) = sqrt(mean(var(X,0,2)));
    % Compute CRPS at each point
    for jj=1:40
        FCRPS(jj,ii) = getCRPS(X(jj,:),ones(Ne,1)/Ne,XT(jj,ii));
    end
    % Get obs ensemble
    obs_err = obsErr*randn(40,Ne);
    obs_err = obs_err - mean(obs_err,2);
    YN = exp(0.5*abs(X-2.5) + obs_err);
    % Prior inflation:
    X = FM(:,ii) + rInf*(X - FM(:,ii));
    % EnKF:
    C = cov([X;YN]');
    CXY = L.*C(1:40,41:80); % Localize
    CYY = L.*C(41:80,41:80); % Localize
    X = X + CXY*(CYY\(Y(:,ii) - YN));
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
