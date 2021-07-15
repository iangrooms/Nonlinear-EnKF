%% Implements EnKF for L96.
% Applies Gaussian Anamorphosis via piecewise-linear 
%clear all, close all
% Set parameters
%Ne = 100; % Ensemble size
%locRad = 4; % Localization radius
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
    YN = 1./(1 + exp(0.5*(X-2.5) + obs_err));
    sx = std(X,0,2);
    my = mean(YN,2);
    sy = std(YN,0,2);
    % Transform everything
    for k=1:40
        [~,I] = sort(X(k,:));
        XHat(k,I) = ZHat0;
        [~,I] = sort(YN(k,:));
        YNHat(k,I) = ZHat0;
        yHat(k) = interp1([0 YN(k,I) 1],[-20 ZHat0 20],Y(k,ii));
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
