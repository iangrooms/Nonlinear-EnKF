function out = lognormal_pdf(t,mu,sigma)
% Evaluates lognormal pdf at t
    out = (1./(t*sigma*sqrt(2*pi))).*exp(-0.5*((log(t)-0.5*abs(mu-2.5))/sigma).^2);
end