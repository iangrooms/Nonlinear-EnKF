locRad = 1;
rInf = 1.05;
Ne = 120;
dtObs = 0.05;
Nt = 5500; % Number of assimilation cycles
obsErr = 1; % obs error
% Get reference data and obs
rng(0) % ensure reproducibility
[T,XT] = ode45(@RHS,[0 linspace(9,9+(Nt-1)*dtObs,Nt)],randn(40,1));
XT = XT(2:end,:)';T = T(2:end);
Y = XT + obsErr*randn(size(XT)); % Add obs error
rng(1);
X0 = bsxfun(@plus,XT(:,1),randn(40,200));

% Set up localization
x = [0:20 -19:-1]';
loc = exp(-.5*(x/locRad).^2);
L = zeros(40);
for ii=1:40
    L(:,ii) = circshift(loc,[ii-1 0]);
end
clear x

% Initialize ensemble
X = X0(:,1:Ne);

% Allocate space for results
FM = NaN(40,Nt); % Forecast mean
FS = NaN(Nt,1); % Forecast spread
FCRPS = FM; % Forecast CRPS
AM = FM; % Analysis mean
AS = NaN(Nt,1); % Analysis spread
ACRPS = AM; % Analysis CRPS

rng(2); % For reproducibility
