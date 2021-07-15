%% Implements EnKF for L96.
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
        LN = logitnormal_pdf(Y(k,ii),X(k,:),obsErr);
        post_pdf = RHF_PDF2(ZN,LN);
        [~,I] = sort(ZN);
        post_cdf = @(t) integral(post_pdf,-Inf,t,'WayPoints',ZN(I));
        parfor kk=1:Ne
            ZNA(kk) = fzero(@(t) post_cdf(t) - kk/(Ne+1),ZN(I(kk)),opts);
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

function out = logitnormal_pdf(t,mu,sigma)
% Evaluates logitnormal pdf at t
    out = (1/(t*(1-t)*sigma*sqrt(2*pi)))*exp(-0.5*((log(t/(1-t))+0.5*(mu-2.5))/sigma).^2);
end
