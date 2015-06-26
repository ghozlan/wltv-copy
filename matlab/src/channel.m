%% CHANNELS

T_SIMULATION = SIM.T_SIMULATION
dt = SIM.dt
df = SIM.df

T_sym = 1/W
%% Additive White Gaussian Nosie (AWGN) Channel
%
% Consider an AWGN channel with
% bandwidth $W$ (Hz),
% power constraint $P$ (Watts) and
% noise (one-sided) power spectral density is $N_0$ (Watts/Hz).
%
% The capacity $C$ in bits/sec is given by
%
% $$C = W \log\left(1 + \frac{(P/W)}{N_0} \right) $$
%

%set(figure(2011),'Name','Mutual Info [Baseline]')
PdB_vec = 0:20;
P_vec = 10.^(PdB_vec/10);

%I0 = (1/T_sym) * log(1+P_vec);
R_AWNG = W * log(1+P_vec/W/(N0));
figure(2000)
plot(P_vec,R_AWNG,'k--x','LineWidth',2,'MarkerSize',10)
legend('AWGN (formula)')

%% Linear Time-Varying (LTV) Channel
% The output $u(t)$ of a linear time-varying (LTV) channel for input
% $x(t)$ can be written as:
%
% $$u(t) = \int h(t,\tau) x(t-\tau) d\tau $$
%
% Equivalently (using the kernel representation)
%
% $$u(t) = \int k_0(t,\tau) x(\tau) d\tau $$
%
% where $k_0(t,\tau) = h(t,t-\tau)$

%% Narrowband Linear Time-Varying (LTV) Channel
% In the ``narrowband case'' case, the output signal can be written as
%
% $$u(t) = \int\int H(\nu,\tau) ~ e^{j 2\pi \nu t} x(t-\tau) ~ d\nu d\tau$$
%
% where $H(\nu,\tau) = \mathcal{F}_t\{h(t,\tau)\}$ 
% is the narrowband spreading function.


%% Widebandband Linear Time-Varying (LTV) Channel
% In the ``wideband'' case, the output signal can be written as
%
% $$u(t) = \int\int h_{WB}(\alpha,\tau) \sqrt{\alpha} x(\alpha(t-\tau)) ~ d\alpha d\tau $$
%
% where $h_{WB}(\alpha,\tau)$ is the wideband spreading function.
% 
% It is more convenient for numerical simulation to have the kernel representation for the channel.
% By doing a transformation of variables $\alpha(t-\tau) = t-\tau^\prime$. We now have
%
% $$u(t) = \int 
% \left[\int \chi\left(\alpha,t-\frac{t-\tau^\prime}{\alpha}\right) \frac{d\alpha}{\sqrt{\alpha}} \right]  
% x(t-\tau^\prime) ~ d\tau^\prime$$
%
% Recognize that
%
% $$h(t,\tau) = \int \chi\left(\alpha,t-\frac{t-\tau}{\alpha}\right) \frac{d\alpha}{\sqrt{\alpha}}$$
%
% hence we have
%
% $$h(t,t-\tau) = \int \chi\left(\alpha,t-\frac{\tau}{\alpha}\right) \frac{d\alpha}{\sqrt{\alpha}}$$
%
% We consider channel with a finite number of paths:
%
% $$h_{WB}(\alpha,\tau) = \sum_m h_m \delta(\alpha-\alpha_m) \delta(\tau-\tau_m)$$
%
% or equivalently
%
% $$h(t,\tau) = \sum_m h_m \sqrt{\alpha_m} \delta(\tau-\tau_m(t))$$ 
%
% where $\tau_m(t) = \alpha_m(\tau_m-t) + t$.
%
% Write $\tau_m(t) = \alpha_m \tau_m - (\alpha_m-1) t$. We have 
% [the expression used in the simulation]
%
% $$h(t,t-\tau) = \sum_m h_m \sqrt{\alpha_m} \delta(t-\tau-\tau_m(t))$$ 
%
% Write $\tau_m(t) = t - \alpha_m(t-\tau_m)$.
%
% $$h(t,t-\tau) 
% = \sum_m h_m \sqrt{\alpha_m} \delta(t-\tau-\tau_m(t))
% = \sum_m h_m \sqrt{\alpha_m} \delta(\tau-t+\tau_m(t))
% = \sum_m h_m \sqrt{\alpha_m} \delta(\tau-\alpha_m(t-\tau_m))$$ 
%

%% Frequency-Selective/Intersymbol Interference (ISI) Linear Time-Invariant (LTI) AWGN Channel
%
% Consider a frequency-selective AWGN channel with
% bandwidth $W$ (Hz),
% power constraint $P$ (Watts),
% noise (one-sided) power spectral density is $N_0$ (Watts/Hz) and
% channel transfer function $H(f)$.
%
% The information rate $R$ in bits/sec is given by
%
% $$R = \int_{<W>} \log\left(1 + \frac{|H(f)|^2 P(f)}{N_0} \right) df$$
%
% where $\int_{<W>} P(f) df \leq P$.
%

% %Hf = fft(H_WLTV(:,1)) .* fft(H_TX(:,1));
% %Hf = fft(H_TX(:,1))/16;
% Hf = fft(H_WLTV(:,1)) .* fft(H_TX(:,1)) / 16;
% % I_vec = [];
% % for P = P_vec
% % I = 1/T_sym * sum( log(1 + abs(Hf).^2 * P) ) * df;
% % I_vec = [I_vec I];
% % end
% % I_LTI = I_vec;

%CHANNEL = 0;
N_FFT = T_SIMULATION/dt;
f = linspace(-F_samp/2,F_samp/2,N_FFT);
W = B_TOTAL;

% if(CHANNEL==ISI) % =============================ISI========================
Hf_AWGN = fft(H_TX(:,1)) * dt;  % DOESNOT GIVE RIGHT ANSWER
Hf_AWGN = fft(H_TX(:,1)) * dt * sqrt(W);
Hf_AWGN = fftshift( (abs(f(:)-fc)<W/2) );
%I_AWGN = sum( log(1 + abs(Hf_AWGN).^2 * (P_vec/W) / N0) ) * df;
R_AWGN = sum( log(1 + abs(Hf_AWGN).^2 * (P_vec/W) / N0) ) * df;
figure(2000)
hold on
plot(P_vec,R_AWGN,'bo','LineWidth',3)
hold off
legend('AWGN (formula)','AWGN (using ISI formula)')


%Hf_ISI = fft(H_WLTV(:,1)) .* fft(H_TX(:,1)) / 16;
Hf_ISI = fft(H_TX(:,1)) .* fft(H_ISI(:,1))*dt;
Hf_ISI = Hf_AWGN .* fft(H_ISI(:,1))*dt;
% %I_LTI = 1/T_sym * sum( log(1 + abs(Hf_ISI).^2 * P_vec) ) * df;
% I_LTI = sum( log(1 + abs(Hf_ISI).^2 * (P_vec/W) / N0) ) * df;
I_vec = [];
for P = P_vec
Pf = zeros(length(Hf_ISI),1);
Pf(abs(f-fc)<W/2) = P/W;
%[sum(Pf)*df P]
I = sum( log(1 + abs(Hf_ISI).^2 * (P/W) / N0) ) * df;
%I = sum( log(1 + abs(Hf_ISI).^2 .* Pf / N0) ) * df;
I_vec = [I_vec I];
end
I_LTI = I_vec;
hold on
plot(P_vec,I_LTI,'r+','LineWidth',2)
hold off
legend('AWGN (formula)','AWGN (using ISI formula)','ISI (formula)')

set(figure(7001),'Name','H(f) AWGN vs ISI')
subplot(2,1,1)
%imagesc(abs(Hf_AWGN*P_vec))
plot(abs(Hf_AWGN))
title('H(f) for AWGN channel')
subplot(2,1,2)
%imagesc(abs(Hf_ISI*P_vec))
plot(abs(Hf_ISI))
title('H(f) for ISI channel')

set(figure(7002),'Name','H(f) AWGN vs ISI')
plot(f,fftshift(abs(Hf_AWGN)),'b'); 
hold on; 
plot(f,fftshift(abs(Hf_ISI)),'r'); 
plot(f,abs( (1-exp(-i*2*pi*f*1.5*T_sym)) ) .* (abs(f-fc)<W/2),'k--','LineWidth',2)
%plot(f,abs( (1-exp(-i*2*pi*f*1.5*T_sym)) ) .* (abs(f-fc)<W/2)*1/sqrt(W),'k--','LineWidth',2)
axis([-1 1 0 3])
hold off
xlabel('f')
ylabel('|H(f)|')
title('H(f) for  AWGN channel vs. ISI (frequency-selective) channel')
legend('AWGN','ISI')

figure(7003);
p_idx = ceil(1);
J1 = 1/T_sym * ( log(1 + abs(Hf_AWGN).^2 * P_vec(p_idx)) ) * df;
J2 = 1/T_sym * ( log(1 + abs(Hf_ISI).^2 * P_vec(p_idx)) ) * df;
freq  = fftshift(f);
plot(freq,J1,'b',freq,J2,'r--')

figure(7004)
plot(t,real(H_ISI(:,1)),'r','LineWidth',2)
xlabel('t'); 
ylabel('h(t)');
title('Impulse response')
axis([-1 10 -2/dt 2/dt])

% % figure(7005)
% % subplot(1,2,1);imagesc(abs((H_ISI)));axis square;
% % subplot(1,2,2);imagesc(abs(fft(H_ISI))); axis square;
% % colormap(gray)
% figure(50)
% plot(f,fftshift(abs(Hf_AWGN)),'b'); 
% hold on; 
% plot(f,fftshift(abs(Hf_ISI)),'r','LineWidth',2); 
% plot(f,abs( (1-exp(-i*2*pi*f*1.5*T_sym)) ) .* (abs(f-fc)<W/2),'k--','LineWidth',2)
% %plot(f,abs( (1-exp(-i*2*pi*f*1.5*T_sym)) ) .* (abs(f-fc)<W/2)*1/sqrt(W),'kx','LineWidth',2)
% hold off
% axis([-1 1 0 3])

%% Power Allocation (Waterfilling) for LTI AWGN Channel
%
% The optimal power allocation $P_{opt}(f)$ is
%
% $$P_{opt}(f) = \max\left(\frac{1}{\lambda} - \frac{N_0}{|H(f)|^2}, 0 \right)$$
%
% where $\lambda$ is such that 
%
% $\int_{<W>} P_{opt}(f) df = P$.

p_idx = 6;
P_max = P_vec(p_idx)

Pf = waterfill(Hf_ISI,P_max/df);

Hf2_ISI = abs(Hf_ISI).^2;
set(figure(21),'Name','Waterfilling')
plot(f,fftshift(1./Hf2_ISI),'r',f,fftshift(Pf+1./Hf2_ISI),'k--','LineWidth',2)
xlabel('f')
ylabel('N_0/|H(f)|^2')
P_WF = sum(Pf) * df %XXXXXXXXXXXXXXXXXXXXXXXX SOMETHING WRONG
title(['P_{max} = ' num2str(PdB_vec(p_idx)) ' dB, P = ' num2str(10*log10(P_WF)) ' dB'])
axis([-2 2 0 10])
legend('N_0/|H(f)|^2','N_0/|H(f)|^2+P(f)')

set(figure(22),'Name','PA using waterfilling')
plot(f,fftshift(Hf2_ISI),'b',f,fftshift(Pf),'k--','LineWidth',2)
xlabel('f')
title('Power allocation using waterfilling')
legend('|H(f)|^2','P(f)')
axis([-2 2 0 10])

%% Information Rate with Waterfilling Power Allocation
I_vec = [];
for P = P_vec
Pf = waterfill(Hf_ISI,P/df);
P_WF = sum(Pf)*df;
I = sum(log(1+ abs(Hf_ISI).^2 .* Pf / N0 )) * df;
I_vec = [I_vec I];
fprintf('P_WF = %.2f, P = %.2f\n',P_WF,P)
end
I_ISI_PA = I_vec;
figure(2000)
hold on
plot(P_vec,I_ISI_PA,'gs:','LineWidth',2)
hold off
legend('AWGN (formula)','AWGN (using ISI formula)','ISI (formula), no waterfilling','ISI (formula), waterfilling')


%% Workbench of waterfilling code
% idx = 6
% P_max = P_vec(idx)
% 
% eta = 0.5;
% k = 1;
% lambda = max(Hf2);
% while(true)
% Pf = max( 1/lambda - 1./Hf2, 0);
% P_current = sum(Pf) * df
% gap = P_max - P_current;
% % if P_much smaller than P_max, allocate more power by raising the water level (1/lambda)
% if (gap > P_max * 1e-5) 
% %lambda = lambda - eta;
% lambda = lambda * (1 - eta * gap/P_max); k = k+1;
% %lambda = lambda * (1 - (eta)^k * gap/P_max); k = k+1;
% %lambda = 1/(lambda + 1);
% else
%     break
% end
% 
% figure(21)
% %plot(f,Hf2,'b--',f,1./Hf2,'r','LineWidth',2)
% plot(f,fftshift(1./Hf2),'r',f,fftshift(Pf+1./Hf2),'k--','LineWidth',2)
% xlabel('f')
% ylabel('N_0/|H(f)|^2')
% P_WF = sum(Pf) * df;
% title(['P_{max} = ' num2str(PdB_vec(idx)) ' dB, P = ' num2str(10*log10(P_WF)) ' dB'])
% axis([-2 2 0 10])
% legend('N_0/|H(f)|^2','N_0/|H(f)|^2+P(f)')
% %pause(0.25)
% end
% 
% display('done')

% end % =============================ISI=====================================
