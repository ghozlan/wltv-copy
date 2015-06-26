%% waterfill(Hf,P)
function Pf = waterfill(Hf,P_max)
df = 1;
Hf2 = abs(Hf).^2;

eta = 0.5; % parameter controlling the rate of raising the "water level"

% 1/lambda is the "water level"
% this initial value of water level implies that the initial total power is zero
lambda = max(Hf2); 
while(true)
    Pf = max( 1/lambda - 1./Hf2, 0);
    P_current = sum(Pf) * df;
    gap = P_max - P_current;
    % if P_much smaller than P_max, allocate more power by raising the water level (1/lambda)
    if (gap > P_max * 1e-5) 
        lambda = lambda * (1 - eta * gap/P_max); 
    else
        break
    end
end