%% Implements EnKF for L96.
% Applies Gaussian Anamorphosis via piecewise-linear 
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
    YN = exp(0.5*abs(X-2.5) + obs_err);
    sx = std(X,0,2);
    my = mean(YN,2);
    sy = std(YN,0,2);
    % Transform everything
    for k=1:40
        [~,I] = sort(X(k,:));
        XHat(k,I) = ZHat0;
        [~,I] = sort(YN(k,:));
        YNHat(k,I) = ZHat0;
        yHat(k) = interp1([0 YN(k,I) (my(k)+4*sy(k))],...
                           [-20 ZHat0 4],Y(k,ii),'linear',4);
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
        X(k,:) = interp1([-4 XHat0(k,:) 4],...
            [(FM(k,ii)-4*sx(k)) X(k,:) (FM(k,ii)+4*sx(k))],XHat(k,:),...
            'linear',FM(k,ii));
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
