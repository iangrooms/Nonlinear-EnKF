%% Implements EnKF for L96.
setup

% Run filter
tmp = (1:Ne)/(Ne+1);
ZHat0 = norminv(tmp);
clear tmp
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
    YN = X + obs_err;
    sx = std(X,0,2);
    sy = std(YN,0,2);
    % Transform everything
    for k=1:40
        [~,I] = sort(X(k,:));
        XHat(k,I) = ZHat0;
        [~,I] = sort(YN(k,:));
        YNHat(k,I) = ZHat0;
        yHat(k) = interp1([(FM(k,ii)-10*sy(k)) YN(k,I) (FM(k,ii)+10*sy(k))],...
                           [-10 ZHat0 10],Y(k,ii));
    end
    XHat0 = XHat; % Save for later
    % Prior inflation in transformed space:
    XHat = mean(XHat,2) + rInf*(XHat - mean(XHat,2));
    % EnKF within transformed space:
    C = cov([XHat;YNHat]');
    CXY = L.*C(1:40,41:80); % Localize
    CYY = L.*C(41:80,41:80); % Localize
    XHat = XHat + CXY*(CYY\(yHat' - YNHat));
    % Transform back
    parfor k=1:40
        X(k,:) = interp1([-20 XHat0(k,:) 20],...
            [(FM(k,ii)-20*sx(k)) X(k,:) (FM(k,ii)+20*sx(k))],XHat(k,:));
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
