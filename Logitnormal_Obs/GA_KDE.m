%% Implements EnKF for L96.
% Applies Gaussian Anamorphosis via MATLAB's kernel density estimation
%clear all, close all
% Set parameters
%Ne = 100; % Ensemble size
%locRad = 3; % Localization radius
%rInf = 1.1; % Ensemble perturbations multiplied by rInf before assimilation

% Configure observing system
%dtObs = 0.05; % Observation window; 0.2 = one day
%Nt = 5500; % Number of assimilation cycles
%obsErr = 1; % obs error

% Get reference data and obs
%rng('shuffle') % ensure different initial seeds for each run
%[T,XT] = ode45(@RHS,[0 linspace(9,9+(Nt-1)*dtObs,Nt)],randn(40,1));
%XT = XT(2:end,:)';T = T(2:end);
%Y = 1./(1 + exp(0.5*(XT-2.5) + obsErr*randn(size(XT))));
%Y(Y==1) = 1 - 1E-10; % ksdensity can't handle points on the boundary
%Y(Y==0) = 1E-10;     % of the support. These only exist due to roundoff anyways

% Set up localization
x = [0:20 -19:-1]';
loc = exp(-.5*(x/locRad).^2);
L = zeros(40);
for ii=1:40
    L(:,ii) = circshift(loc,[ii-1 0]);
end
clear x

% Initialize ensemble
rng(1); % For reproducibility
X = bsxfun(@plus,XT(:,1),obsErr*randn(40,Ne));

% Allocate space for results
FM = NaN(40,Nt); % Forecast mean
FS = NaN(Nt,1); % Forecast spread
FCRPS = FM; % Forecast CRPS
AM = FM; % Analysis mean
AS = NaN(Nt,1); % Analysis spread
ACRPS = AM; % Analysis CRPS

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
    YN = 1./(1 + exp(0.5*(X-2.5) + obs_err));
    YN(YN==1) = 1 - 1E-10;
    YN(YN==0) = 1E-10;
    % Transform everything
    XHat = X;
    YNHat = YN;
    yHat = Y(:,ii);
    parfor k=1:40
        [kde,~] = ksdensity(X(k,:),X(k,:),'Function','cdf');
        XHat(k,:) = norminv(kde(:)');
        [kde,~] = ksdensity(YN(k,:),[YN(k,:) Y(k,ii)],'Function','cdf',...
            'support',[0 1]);
        YNHat(k,:) = norminv(kde(1:end-1)');
        yHat(k) = norminv(kde(end));
    end
    % Prior inflation in transformed space:
    XHat = mean(XHat,2) + rInf*(XHat - mean(XHat,2));
    % EnKF within transformed space:
    C = cov([XHat;YNHat]');
    CXY = L.*C(1:40,41:80); % Localize
    CYY = L.*C(41:80,41:80); % Localize
    XHat = XHat + CXY*(CYY\(yHat - YNHat));
    % Transform back
    parfor k=1:40
        [kde,~] = ksdensity(X(k,:),normcdf(XHat(k,:)),'Function','icdf');
        X(k,:) = kde;
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
