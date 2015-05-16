function layer = layer_indices( SCHEME, SIM, IS_RX )

if nargin == 2
    IS_RX = false; 
end

T_XX = SIM.T_TRANSMISSION;
if IS_RX==true, T_XX = SIM.T_RX; end

W_base = SCHEME(1); a_base = SCHEME(2); K_prime = SCHEME(3);

W_vec = W_base * a_base.^(0:K_prime-1);
N_symb_per_layer = floor(T_XX .* W_vec)*2;
N_sym_total = sum(N_symb_per_layer);

first_index_of_layer = cumsum([0 N_symb_per_layer(1:end-1)]) + 1;
last_index_of_layer = cumsum(N_symb_per_layer(1:end));

for layer_index = 1:K_prime
    layer{layer_index} = first_index_of_layer(layer_index):last_index_of_layer(layer_index);
end

end

