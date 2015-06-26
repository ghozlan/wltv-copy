AWGN = 1; ISI = 2; NBLTV = 3; WBLTV = 4;
CHANNEL = AWGN;
CHANNEL = ISI;
%CHANNEL = WBLTV;

tic
%% Simulation Paramters
PASSBAND=false;
T_TRANSMISSION=32;
F_samp=64;
N0=1;                   % one-sided power spectral density of noise

SIM.PASSBAND = PASSBAND;
if(PASSBAND)
    SIM.REAL_DIM_PER_SYM = 1;
else
    SIM.REAL_DIM_PER_SYM = 2;
end

SIM.T_TRANSMISSION = T_TRANSMISSION;
SIM.T_SIMULATION = 3*T_TRANSMISSION;
SIM.df = 1.0/SIM.T_SIMULATION;

SIM.F_samp = F_samp;
SIM.dt = 1.0/F_samp;

t = linspace(0,SIM.T_SIMULATION,SIM.T_SIMULATION/SIM.dt+1);
t(end) = [];
SIM.t = t;

SIM.N0 = N0;

%% Transmission Scheme Parameters

% SCHME format: [W (base), a (base), K_prime, fc (base)]
SCHEMES{1} = [1,2,3,1.5]; %fc_base = 1.5*W_base;
SCHEMES{2} = [1,1.587401051968199,4,1.5]; %fc_base = 1.5*W_base;
SCHEMES{3} = [7,1,1,4.5];

W = 1 % Bandwidth
SCH = [W, 1, 1, 0];

W_base = SCH(1); a_base = SCH(2); K_prime = SCH(3); fc_base = SCH(4);
fprintf('W = %f, a = %f, K^prime = %d\n', W_base, a_base, K_prime)

%% Generate the tx vectors

fprintf('Generating transmitter matrix\n')
[H_TX f_min f_max] = generate_vecs(W_base,a_base,K_prime,fc_base, SIM);
B_TOTAL = f_max - f_min
fc = (f_max + f_min)/2

%return
%% Plot the transmit vectors (columns of H_TX) in time domain and frequecny domain

%plot_tx( H_TX )

%% Channel Parameters

% Deterministic channels

% Wideband LTV Channels
if(CHANNEL==AWGN)
    CH.N_paths = 1; CH.h_wb = 1; CH.tau = 0; CH.alpha = 1; 
end

% Linear time-invariant channels (intersymbol intereference/frequency-selective channels)
CH.N_paths = 2; CH.h_wb = [1 1/2]; CH.tau = [0 2]; CH.alpha = [1 1];
ISI_CHANNELS{1} = CH;
CH.N_paths = 2; CH.h_wb = [1 2]; CH.tau = [0 2]; CH.alpha = [1 1];
ISI_CHANNELS{2} = CH;
CH.N_paths = 2; CH.h_wb = [1 -1]; CH.tau = [0 1.5]; CH.alpha = [1 1];
ISI_CHANNELS{3} = CH;

if(CHANNEL==ISI)
    CH = ISI_CHANNELS{channel_index};
end

% Wideband linear time-varying channels
CH.N_paths = 2; CH.h_wb = [1 1/2]; CH.tau = [0 2]; CH.alpha = [1 2]; %CHANNEL A
WBLTV_CHANNELS{1} = CH;
CH.N_paths = 2; CH.h_wb = [1 1/2]; CH.tau = [0 2]; CH.alpha = [1 1.587401051968199]; % CHANNEL B
WBLTV_CHANNELS{2} = CH;
CH.N_paths = 2; CH.h_wb = [1 1.5]; CH.tau = [0 2]; CH.alpha = [1 2]; %CHANNEL C
WBLTV_CHANNELS{3} = CH;
CH.N_paths = 2; CH.h_wb = [1 1.5]; CH.tau = [2 3]; CH.alpha = [1 2]; %CHANNEL D
WBLTV_CHANNELS{4} = CH;
CH.N_paths = 3; CH.h_wb = [1 -0.7 1.5]; CH.tau = [2 1 3]; CH.alpha = [1 1.25 2]; %CHANNEL E
WBLTV_CHANNELS{5} = CH;

CHANNEL_LABEL = {'A','B','C','D','E'};

if(CHANNEL==WBLTV)
    CH = WBLTV_CHANNELS{channel_index};
end


%% Generate the channel matrix

fprintf('Generating channel matrix\n')

K0_t_tau = generate_ch_matrix(CH, SIM);

H_CH = K0_t_tau * SIM.dt;

%% Generate the rx vectors

H_RX = H_TX * SIM.dt;

%% Transmission matrix, Channel matrix, Receiver matrix and Spectrum of Input vs Spectrum of Output

%plot_txchrx(H_TX, H_CH, H_RX);

%% Unforgiven

if(CHANNEL==ISI)
    H_ISI = K0_t_tau;
    channel
end

display('done')