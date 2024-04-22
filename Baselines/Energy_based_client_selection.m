% Author: Jingjing Zheng
% work address:
% CISTER Research Centre, ISEP, Polytechnic Institute of Porto (IPP) 
% Department of Electrical and Computer Engineering, Faculty of Engineering, University of Porto, Porto, Portugal
% email: zheng@isep.ipp.pt
% November 2020; Last revision: 12-December-2020
%%%%%% Client selection based on energy sequence

clc
clear all
close all
tic
%%%%  Initialization and server access to client information, collect client's information 
 load ('./../Original_data_infomation/original_data_1000_20.mat');
% load ('./../Original_data_infomation/original_data_1000_40.mat');
%  load ('./../Original_data_infomation/original_data_1000_60.mat');
%  load ('./../Original_data_infomation/original_data_1000_80.mat');
%  load ('./../Original_data_infomation/original_data_1000_100.mat');

All_clients_dataset_info; %each client dataset in each epoch
All_clients_bandwidth_info;  %each client bandwidth
All_clients_transmission_power_info;
G; % each client channel gain 
f; % each client frequency
unit_cost;  % each data cost   unit:  cycles/bit 

ratio = 0;
[T_round, Number_of_clients] = size(All_clients_dataset_info);  % T_round: the total number of t_round or iteration; Number_of_clients:  the number of clients
each_t_round_ratio = [];
t_round_value = [];
each_t_round_index = [];
%All_clients_dataset_info = unifrnd(500, 1000, T_round, Number_of_clients)*1.0e+06;  % all client's data size info, which follow uniform distribution [500,1000]MB 1MB = 1.0e+06 byte
mu = 1.7e-08; % system parameter
 B = 1.0e+06;% total bandwidth. 10MHz   unit: Hz
% B = 3.0e+06;
% B = 5.0e+06;
% B = 7.0e+06;
%  B = 9.0e+06

xi = 1.0e-28;
N0 = 1.0e-08; % Channel noise    unit dBm/Hz
T_max = 5;  % T_round  unit: s                     %global_target_accuracy = 1.0e-03; 
S  = 100; % upload or transmit datasize  S = 100 kbits
The_num_of_iters_each_epoch = 10;  % the number of global iterations in each epoch   B 
% The_num_of_iters_each_epoch = 30;
% The_num_of_iters_each_epoch = 50;
% The_num_of_iters_each_epoch = 60;
% The_num_of_iters_each_epoch = 90;

The_num_of_local_iters_each_global_iter = 4; % the number of local iterations in each global iteration  A 
% log(2.718) = 0.9999


ratio = 0;
T_round = 1000;  % the total number of t_round or iteration
each_t_round_ratio = [];
Energy_based_cumulative_t_round_ratio = [];
each_t_round_index = [];
obj_store = [];


for t_round = 1 : 1 : T_round 
    D = All_clients_dataset_info(t_round, :); 
    b = All_clients_bandwidth_info(t_round, :);
    P = All_clients_transmission_power_info(t_round, :);
 
    epsilon_only_one = log(1 +  mu * D);% each user's accuracy.
    epsilon_0 = 0.1; % the lower bound of accuracy
 
 %%%%%%%%% calculate the total energy 
    E_cmp = The_num_of_local_iters_each_global_iter * xi * unit_cost .* D .* f .* f; %Each client's computation power
    E_up =   S * P ./ (b.*log2(1 + (P .* G)./ (N0 * b)));%  Energy consumption of users. P(watt) = 10^(P(dBm)/10) / 1000
    Energy = The_num_of_iters_each_epoch *(E_cmp + E_up);

 %% calculate the total time (delay)
    T = The_num_of_iters_each_epoch * ( The_num_of_local_iters_each_global_iter * unit_cost.* D./f + S./(b.*log2(1 + (P .* G)./ (N0 .* b))));  % time consumption
    

    num = length(P);  % the number of users.
    init_client_select = zeros(1,num); % Initialize the list of client selections
    beta_prime = [];
    i = 1;
    eta = [];
    %beta_prime = [];
    %init_qualified_client_dataset = [];
    init_qualified_client_accuracy = [];
    init_qualified_client_energy = [];
    init_qualified_client_bandwidth = [];
    origin_qualified_client_index = [];
    init_qualified_client_dataset = [];
    origin_unqualified_client_index = [];
    combination_list = zeros(1,10);
    model_accuracy = [];
    f_obj_star = +inf;
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%    Preliminary screening, check bandwidth and time constraint
  for k = 1 : length(f)
    if (b(k) <= B && T(k) <= T_max)
        init_qualified_client_dataset = [ init_qualified_client_dataset, D(k)];
        init_qualified_client_accuracy = [init_qualified_client_accuracy, epsilon_only_one(k)]; % Store the accuracy of qualified clients
        init_qualified_client_energy = [init_qualified_client_energy, Energy(k)];
        init_qualified_client_bandwidth = [init_qualified_client_bandwidth, b(k)];
        origin_qualified_client_index = [origin_qualified_client_index, k];
        init_client_select(k) = 1;
    else
        origin_unqualified_client_index = [origin_unqualified_client_index, k];  % Store the unqualified clients
    end 
  end
  
  if  ~isempty(origin_qualified_client_index)  % if origin_qualified_client_index is non-empty
      energy_based_select = zeros(1,length(origin_qualified_client_index));
      
     %preliminary_qualified_client_select = origin_qualified_client_index;
     preliminary_qualified_client_select_info = [origin_qualified_client_index; init_qualified_client_energy; init_qualified_client_accuracy; init_qualified_client_dataset; init_qualified_client_bandwidth];
     sorted_qualified_client_select = sortrows(preliminary_qualified_client_select_info',4)'; %% sorted qualified client select accroding to eta

    %%%%%%%%%%%%%%%    output unqualified clients
     origin_preliminary_screening_unqualified_client_index = origin_unqualified_client_index;  %%%%************ need to be concatenated
     origin_preliminary_screening_unqualified_client_index(:,:) = 0; % The clients who do not meet the time constraints are assigned a value of 0
     beta_prime = origin_preliminary_screening_unqualified_client_index;   %%%%%%%%%%%%%%%%%% need to be concatenated
    %%%%  s = find(init_client_select) % store the index of selected clients

    %%%%%%%%%%%%%%%%%%%%%%%%%%     client initialization and sort 
     origin_sorted_qualified_client_select_index = sorted_qualified_client_select(1,:);    %%%%************
     sorted_qualified_client_select(1,:) = 0; % Initialize sorted_qualified_client index
     init_client_index = sorted_qualified_client_select(1,:);  %%%%% need to be concatenated
     client_energy = sorted_qualified_client_select(2,:);
     client_accuracy = sorted_qualified_client_select(3,:);
     client_dataset = sorted_qualified_client_select(4,:);
     client_bandwidth = sorted_qualified_client_select(5,:);
     obj = [];
     All_select = [];
     qualified_selection = [];
     beta_star = zeros(1, length(init_client_index));
     
%%%%%%%%%%%%%%%%  Select the top n users with the lowest energy consumption
     alternative_select =  zeros(1,length(origin_qualified_client_index));
     top_n = 2; %Select the top n users with the lowest energy consumption
     while( top_n <= length(init_client_index) && 1 <= top_n)
          for i =1: top_n      % Randomly select length(L)*fraction users
                alternative_select(i) = 1;
          end 
          total_bandwidth = alternative_select *  client_bandwidth'; 
          if(total_bandwidth <= B)  %%%  Check the bandwidth
                 % disp("This client selection is qualified");
                  f_obj_star = sum(alternative_select * client_energy') / log(1+ mu*sum( alternative_select * client_dataset')); % Calculate the objective function
                  obj_store = [obj_store, f_obj_star]; % Store the objective function value 
                   beta_star = alternative_select;
                   break;
          else
               for i =1: top_n      % Randomly select length(L)*fraction users
                    alternative_select(i) = 0;
               end               
               top_n = top_n - 1;
          end       
     end     
%      %%%%%combination_list = zeros(1,length(eta)); % initialize the client selection list.
%      while (i <= length(client_eta))
%          if (client_bandwidth(i) <= B) 
%                 f_obj = client_eta(i);  % output the optimal value of objective function 
%                 init_client_index(i) = 1;
%                 beta_star = init_client_index;
%                 break;
%          else
%              i = i + 1;
%          end
%      end     
     origin_index = cat(2,  origin_sorted_qualified_client_select_index, origin_unqualified_client_index)
     final_client_client_selection_index = cat(2, alternative_select, beta_prime) % Ultimate optimal client selection strategy
     if (f_obj_star == +inf)
            disp('There is no qualified client selection because f_obj is invalid');    
     end  
  else
           disp('There is no qualified client selection because origin_qualified_client_index is empty');  
 end  
  ratio = ratio + f_obj_star;
  t_round_value = [t_round_value, f_obj_star]; 
  each_t_round_index = [each_t_round_index, t_round];
  Energy_based_cumulative_t_round_ratio = [Energy_based_cumulative_t_round_ratio,  ratio];
end
toc   
 save  original_data_1000_20_energy_based_result  t_round_value Energy_based_cumulative_t_round_ratio% save variable/vector  cumulative_t_round_ratio to y1data.mat
% save  original_data_1000_40_energy_based_result  t_round_value Energy_based_cumulative_t_round_ratio
% save  original_data_1000_60_energy_based_result  t_round_value Energy_based_cumulative_t_round_ratio
% save  original_data_1000_80_energy_based_result  t_round_value Energy_based_cumulative_t_round_ratio
% save  original_data_1000_100_energy_based_result t_round_value Energy_based_cumulative_t_round_ratio


