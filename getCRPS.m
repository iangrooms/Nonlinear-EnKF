function CRPS = getCRPS(x,w,xt)
[x,I] = sort(x);x=x(:);
w = w(I);w = w(:);
if xt < x(1)
    CRPS = sum(((1-cumsum(w(1:end-1))).^2).*diff(x));
    CRPS = CRPS + (x(1)-xt);
    return
end
ii=1;
CRPS = 0;
while ( (x(ii+1) < xt) && (ii+1 <= length(x)) )
    CRPS = CRPS + (x(ii+1)-x(ii))*sum(w(1:ii))^2;
    ii = ii+1;
    if ( ii == length(x) )
      CRPS = CRPS + (xt-x(end));
      return
    end
end
if (ii < length(x))
    CRPS = CRPS + (xt-x(ii))*sum(w(1:ii))^2;
    CRPS = CRPS + (x(ii+1)-xt)*(1-sum(w(1:ii)))^2;
end
ii = ii+1;
while ( ii < length(x) )
    CRPS = CRPS + (x(ii+1) - x(ii))*(1-sum(w(1:ii)))^2;
    ii = ii+1;
end
