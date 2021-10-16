% Author: Jingjing Zheng
% work address:
% CISTER Research Centre, ISEP, Polytechnic Institute of Porto (IPP) 
% Department of Electrical and Computer Engineering, Faculty of Engineering, University of Porto, Porto, Portugal
% email: zheng@isep.ipp.pt
% November 2020; Last revision: 12-December-2020
%%%%%% Server obatined information

clc
clear all
close all

%%%%  Initialization and server access to user information, collect user's information 
ratio = 0;
T_round = 1000;  % the total number of t_round or iteration
Number_of_clients = 20; % the number of clients
All_clients_dataset_info = unifrnd(2, 10, T_round, Number_of_clients)*1.0e+06;  % all client's data size info, which follow uniform distribution [2,10]MB 1MB = 1.0e+06 byte
All_clients_bandwidth_info = unifrnd(5, 10, T_round, Number_of_clients)*1.0e+04;%*1.0e+04; % all client's bandwidth info, which follow uniform distribution [50,100] unit: KHz
All_clients_transmission_power_info = unifrnd(4, 10, T_round, Number_of_clients); %unit dBm

mu = 1.7e-08; % system parameter
B = 1.0e+06;% total bandwidth.   unit: MHz
%N0 = -174;  % Channel noise    unit dBm/Hz
N0 =  1.0e-08; %unit W/Hz
T_max = 5;  % T_round  unit: s                   
S  = 10^4; % upload or transmit datasize  S = 10 kbits

xi = 1.0e-28;
unit_cost =  unifrnd(10, 30, 1, Number_of_clients);  % each data cost   unit:  cycles/bit 
G = 40* (0.5 ./ unifrnd(0.6, 1, 1, Number_of_clients)).^4;
f = unifrnd(1.0,2.0, 1, Number_of_clients)*1.0e+09;  % unifrnd can generate fraction  1GHz = 1.0e+09 Hz

save original_data_1000_20  All_clients_dataset_info All_clients_bandwidth_info All_clients_transmission_power_info G  f  unit_cost

% The_num_of_iters_each_epoch = 10;  % the number of global iterations in each epoch   B 
% The_num_of_local_iters_each_global_iter = 4; % the number of local iterations in each global iteration  A 


% for t_round = 1 : 1 : T_round 
%     D = All_clients_dataset_info(t_round, :); 
%     b = All_clients_bandwidth_info(t_round, :);
%     %b = [6 15 5 13 13 14 5 9  7 13 9 15 7 7 6 6 14 11 11 6]*1.0e+04%normpdf(t_round,85000,10000);% each user's  bandwidth.  unit: Hz  exp(-t_round)  randi([5,15],1,20)
%     P = All_clients_transmission_power_info(t_round, :);
%   
%     %epsilon = 0.043 * log(1 + 138300 * D );% each user's accuracy.
%     epsilon_only_one = log(1 +  mu * D);% each user's accuracy
%     %epsilon_0 = 1 - 1 / t_round;
%     epsilon_0 = 0.5; % the lower bound of accuracy
%     %epsilon_0= 0.9*0.15*log(1+t_round); 
%    
% %%%%%%%%% calculate the total energy 
%     E_cmp = The_num_of_local_iters_each_global_iter * xi * unit_cost .* D .* f .* f; %Each client's computation power
%     E_up =   S * P ./ (b.*log2(1 + (P .* G)./ (N0 * b)));%  Energy consumption of users. P(watt) = 10^(P(dBm)/10) / 1000
%     Energy = The_num_of_iters_each_epoch *(E_cmp + E_up);
% 
%  %% calculate the total time (delay)
%     T = The_num_of_iters_each_epoch * ( The_num_of_local_iters_each_global_iter * unit_cost.* D./f + S./(b.*log2(1 + (P .* G)./ (N0 .* b))));  % time consumption
%     
% end