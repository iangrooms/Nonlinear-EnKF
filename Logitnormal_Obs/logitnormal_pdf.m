function out = logitnormal_pdf(t,mu,sigma)
% Evaluates logitnormal pdf at t
    out = (1/(t*(1-t)*sigma*sqrt(2*pi)))*exp(-0.5*((log(t/(1-t))+0.5*(mu-2.5))/sigma).^2);
end
