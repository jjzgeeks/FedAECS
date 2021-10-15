% Author: Jingjing Zheng
% work address:
% CISTER Research Centre, ISEP, Polytechnic Institute of Porto (IPP) 
% Department of Electrical and Computer Engineering, Faculty of Engineering, University of Porto, Porto, Portugal
% email: zheng@isep.ipp.pt
% November 2020; Last revision: 12-December-2020
%%%%%% Client selection based on heuristic algorithm 
clc
clear all
close all

tic
%%%%  Initialization and server access to client information, collect client's information 
% load ('./../Original_data_infomation/original_data_1000_20.mat');
% load ('./../Original_data_infomation/original_data_1000_40.mat');
%  load ('./../Original_data_infomation/original_data_1000_60.mat');
 load ('./../Original_data_infomation/original_data_1000_80.mat');
% load ('./../Original_data_infomation/original_data_1000_100.mat');

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
cumulative_t_round_ratio = [];
each_t_round_index = [];
temp_t_round = [];
%All_clients_dataset_info = unifrnd(500, 1000, T_round, Number_of_clients)*1.0e+06;  % all client's data size info, which follow uniform distribution [500,1000]MB 1MB = 1.0e+06 byte
mu = 1.7e-08; % system parameter
% B = 1.0e+06;% total bandwidth. 10MHz   unit: Hz
% B = 3.0e+06;
 % B = 5.0e+06;
 B = 7.0e+06;
 % B = 9.0e+06;

xi = 1.0e-28;
N0 = 1.0e-08; % Channel noise    unit dBm/Hz

%  T_max = 2;  % T_round  unit: s 
 T_max = 3;  
%%%%%%%%%%%%%% T_max = 4; 
% T_max = 5;  % T_round  unit: s 
%%%%%%%%%%%%%%%%% T_max = 6;  
%%%% T_max = 7;   
%%%%%%%%%%%%%%%%%%%%%%  T_max = 8;  
% T_max = 9;  


%global_target_accuracy = 1.0e-03; 
S  = 100; % upload or transmit datasize  S = 100 kbits
The_num_of_iters_each_epoch = 10;  % the number of global iterations in each epoch   B 
% The_num_of_iters_each_epoch = 30;
% The_num_of_iters_each_epoch = 50;
% The_num_of_iters_each_epoch = 60;
% The_num_of_iters_each_epoch = 90;
The_num_of_local_iters_each_global_iter = 4; % the number of local iterations in each global iteration  A 

for t_round = 1 : 1 : T_round 
    D = All_clients_dataset_info(t_round, :); 
    b = All_clients_bandwidth_info(t_round, :);
    %b = [6 15 5 13 13 14 5 9  7 13 9 15 7 7 6 6 14 11 11 6]*1.0e+04%normpdf(t_round,85000,10000);% each user's  bandwidth.  unit: Hz  exp(-t_round)  randi([5,15],1,20)
    P = All_clients_transmission_power_info(t_round, :);
 
 %%%%%%% Calculate accuracy 
    epsilon_only_one = log(1 +  mu * D);% each user's accuracy.
    epsilon_0 = 0.1; % the lower bound of accuracy
%     epsilon_0 = 0.3; 
%     epsilon_0 = 0.5; 
%     epsilon_0 = 0.7; 
%     epsilon_0 = 0.9; 
     
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
    %combination_list(4) = 1;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%    Preliminary screening, check bandwidth and time constraint
  for k = 1 : length(P)
    if (b(k) <= B && T(k) <= T_max)
        eta = [eta, Energy(k) / epsilon_only_one(k)];
        init_qualified_client_accuracy = [init_qualified_client_accuracy, epsilon_only_one(k)]; % Store the accuracy of qualified clients
        init_qualified_client_energy = [init_qualified_client_energy, Energy(k)];
        init_qualified_client_bandwidth = [init_qualified_client_bandwidth, b(k)];
        init_qualified_client_dataset = [init_qualified_client_dataset, D(k)];
        origin_qualified_client_index = [origin_qualified_client_index, k];
        init_client_select(k) = 1;
    else
        origin_unqualified_client_index = [origin_unqualified_client_index, k];  % Store the unqualified clients
        % init_client_select(k) = 0;
    end 
  end
   %%%%%%%%%%%%%%%    output unqualified clients
    origin_preliminary_screening_unqualified_client_index = origin_unqualified_client_index;  %%%%************ need to be concatenated
    if  ~isempty(origin_unqualified_client_index)
            origin_unqualified_client_index(:,:) = 0; % The clients who do not meet the time constraints are assigned a value of
            beta_prime = origin_unqualified_client_index;
    else
            beta_prime = origin_unqualified_client_index;   %%%%%%%%%%%%%%%%%% need to be concatenated
    end
    %%%%  s = find(init_client_select) % store the index of selected clients
  

  if  ~isempty(origin_qualified_client_index)  % if origin_qualified_client_index is not empty
     origin_qualified_client_index;
    %origin_qualified_client_index(:,:) = 1;
     preliminary_qualified_client_select = origin_qualified_client_index;
     preliminary_qualified_client_select_info = [origin_qualified_client_index; eta; init_qualified_client_accuracy; init_qualified_client_energy; init_qualified_client_bandwidth; init_qualified_client_dataset];
     sorted_qualified_client_select = sortrows(preliminary_qualified_client_select_info',2)'; %% sorted qualified client select accroding to eta

    %%%%%%%%%%%%%%%%%%%%%%%%%%     client initialization
     origin_sorted_qualified_client_select_index = sorted_qualified_client_select(1,:);    %%%%************
     sorted_qualified_client_select(1,:) = 0; % Initialize sorted_qualified_client index
     init_client_index = sorted_qualified_client_select(1,:);  %%%%% need to be concatenated
     client_eta = sorted_qualified_client_select(2,:);
     client_accuracy = sorted_qualified_client_select(3,:);
     client_energy = sorted_qualified_client_select(4,:);
     client_bandwidth = sorted_qualified_client_select(5,:);
     client_dataset = sorted_qualified_client_select(6,:);
     obj = [];
     All_select = [];
     qualified_selection = [];
     beta_star = zeros(1, length(client_eta));
     
   %%%%%combination_list = zeros(1,length(eta)); % initialize the client selection list.
     while (i <= length(client_eta))
        if (client_accuracy(i) >= epsilon_0) %%%  Check accuracy  problem
               % disp('Minimum value of objective function:');
                f_obj_star = client_eta(i);  % output the optimal value of objective function 
                init_client_index(i) = 1;
                beta_star = init_client_index;
                break;
        else
             % li = min(40, length(client_eta) -1)
             % li = min(40, length(client_eta) -1)
            if( i <= length(client_eta) -1)  
                 i = i+1; 
                 while (2 <= i && i <= length(client_eta))
                      if( 20 <= i) %%%%%%% the i is larger than 20, current t_round is stop 
                            temp_t_round = [temp_t_round; t_round];
                            f_obj_star = inf;
                            break;
                      else
                         if(client_accuracy(i) >= epsilon_0) %%  Check accuracy 
                                         disp('ratio value eta_i:');
                                         client_eta(i); 
                                         init_client_index(i) = 1; %% select user i  
                                          The_first_qualified_client_index = init_client_index; % store the_first_qualified_client_index
                           %   beta_star = init_client_index
         %%%%%%%%%%%%%%%% Check the combination selection of  the previous i clients
                                         for t = 1:i-1
                                                s = nchoosek(1:i-1,t);   % Find different combinations of selections for the first i-1 clients
                                                 row = size(s,1); % Calculate the number of rows
                                                select = repmat(init_client_index,row,1); % generate the same number of rows 
                                                for  p = 1:row
                                                    select(p, s(p,:)) = 1; %%%%%%%%%XXXXXXXXXXXXXXXXXXXXXXXXXx
                                                end
                           
                                                All_select = [All_select; select];  % List all possible client selection options
                                  % tt = cat( 1,select,select)
                                          end
                            %total_aacuracy = All_select * client_accuracy'; % Calculate accuracy
                            
                            %%%% Calculate model accurcy
                             %model_accuracy = All_select*client_accuracy' ./ sum(All_select,2); % sum(A,2)  sum the matri A accroding to each row
                                    model_accuracy = log(1+ mu*All_select*client_dataset');
                                    total_bandwidth = All_select* client_bandwidth'; % Calculate the total bandwidth of each option
    
     %%%%%%%%%%%%%%%%%%%  Filter combinations that unsatisfied constraints
                                    for u = 1:size(All_select,1)
                                         if(epsilon_0 <= model_accuracy(u) && total_bandwidth(u) <= B ) %  Check the constraints are whether satisfied
                                             f_obj = sum(All_select(u,:)*client_energy') / model_accuracy(u); % Calculate the objective function
                                             obj = [obj,f_obj]; % Store the objective function value 
                                              qualified_selection = [qualified_selection; All_select(u,:)];  %  Store the qualificated selection
                                         end 
                                     end 
                         
                         %%%%%  Check whether there is a client selection for combinatorial optimization satisfying constraints
                                    if ~isempty(qualified_selection)
                                        disp('The value of objective fucntion list: ')
                                        obj;
                                         [x,y] = find(obj == min(min(obj))); % y is the location of objective function minimum value 
                                         disp('The sub-optimal value is: ')
                                         obj(y)
                                         client_eta(i)
                                         qualified_selection(y,:) 
   
                        % Further compare the optimal values for the objective function
                                         if ( obj(y) <= client_eta(i))
                                             disp('the optimal value of objective function is:')
                                             f_obj_star = obj(y);
                                             beta_star = qualified_selection(y,:); 
                                         else
                                             disp('the optimal value of objective function is:')
                                             f_obj_star = client_eta(i)
                                             beta_star = The_first_qualified_client_index;  
                                         end 
                                    else
                                        disp('the optimal value of objective function is:')
                                        client_eta(i)
                                        beta_star = The_first_qualified_client_index;  
                                    end
                                break;
                         else 
                                    i = i + 1;
                         end  
                      end
                 end
            else   
                break;
            end
         end   
     origin_index = cat(2, origin_sorted_qualified_client_select_index, origin_preliminary_screening_unqualified_client_index)
     final_client_selection_index = cat(2, beta_star, beta_prime) % Ultimate optimal client selection strategy
     disp('the optimal value of objective function is:')
     f_obj_star
     end
        if (f_obj_star == +inf)
            disp('There is no qualified client selection because f_obj_star is invalid');    
        end 
 else
     disp('There is no qualified client selection because origin_qualified_client_index is empty');  
  end  
    exclude_t_round = unique(temp_t_round); %store the t round that i is too large
    ratio = ratio + f_obj_star
    t_round_value = [t_round_value, f_obj_star]; 
    %tt_round_value = t_round_value(:,exclude_t_round);
    %t_round_value = tt_round_value;
    each_t_round_index = [each_t_round_index, t_round];
    cumulative_t_round_ratio = [cumulative_t_round_ratio, ratio];
   
end
toc 
%save  original_data_1000_20_heuristic_result  t_round_value cumulative_t_round_ratio% save variable/vector  cumulative_t_round_ratio to y1data.mat T_max = 5 seconds
% save  original_data_1000_40_heuristic_result  t_round_value cumulative_t_round_ratio
% save  original_data_1000_60_heuristic_result  t_round_value cumulative_t_round_ratio
% save  original_data_1000_80_heuristic_result  t_round_value cumulative_t_round_ratio
% save  original_data_1000_100_heuristic_result  t_round_value cumulative_t_round_ratio

% save original_data_1000_20_2_heuristic_result  t_round_value cumulative_t_round_ratio % 1000 epochs, 20clients, T_max = 2 seconds
% save original_data_1000_20_3_heuristic_result  t_round_value cumulative_t_round_ratio % T_max = 3 seconds
%  save original_data_1000_20_4_heuristic_result  t_round_value cumulative_t_round_ratio % T_max = 4 seconds
% save original_data_1000_20_6_heuristic_result  t_round_value cumulative_t_round_ratio % 1000 epochs, 20clients, T_max = 6 seconds
% save original_data_1000_20_7_heuristic_result  t_round_value cumulative_t_round_ratio   % 1000 epochs, 20clients, T_max = 7 seconds
% save original_data_1000_20_8_heuristic_result  t_round_value cumulative_t_round_ratio   % 1000 epochs, 20clients, T_max = 8 seconds
% save original_data_1000_20_9_heuristic_result  t_round_value cumulative_t_round_ratio   % 1000 epochs, 20clients, T_max = 9 seconds


%%%%% 1000 epochs, 40 clients, T_max = 2 seconds
%  save original_data_1000_40_2_heuristic_result  t_round_value cumulative_t_round_ratio 
% save original_data_1000_40_3_heuristic_result  t_round_value cumulative_t_round_ratio % T_max = 3 seconds
% save original_data_1000_40_5_heuristic_result  t_round_value cumulative_t_round_ratio %  T_max = 6 seconds
% save original_data_1000_40_7_heuristic_result  t_round_value cumulative_t_round_ratio  %  T_max = 7 seconds
% save original_data_1000_40_9_heuristic_result  t_round_value cumulative_t_round_ratio  %  T_max = 9 seconds


%%%%% 1000 epochs, 60 clients, T_max = 2 seconds
% save original_data_1000_60_2_heuristic_result  t_round_value cumulative_t_round_ratio 
%save original_data_1000_60_3_heuristic_result  t_round_value cumulative_t_round_ratio % T_max = 3 seconds
% save original_data_1000_60_5_heuristic_result  t_round_value cumulative_t_round_ratio %  T_max = 6 seconds
% save original_data_1000_60_7_heuristic_result  t_round_value cumulative_t_round_ratio  %  T_max = 7 seconds
% save original_data_1000_60_9_heuristic_result  t_round_value cumulative_t_round_ratio  %  T_max = 9 seconds

%%%%% 1000 epochs, 80 clients, T_max = 2 seconds
save original_data_1000_80_3_heuristic_result  t_round_value cumulative_t_round_ratio % T_max = 3 seconds
% save original_data_1000_80_5_heuristic_result  t_round_value cumulative_t_round_ratio %  T_max = 6 seconds
% save original_data_1000_80_7_heuristic_result  t_round_value cumulative_t_round_ratio  %  T_max = 7 seconds
% save original_data_1000_80_9_heuristic_result  t_round_value cumulative_t_round_ratio  %  T_max = 9 seconds





%%%%% 1000 epochs, 100 clients, T_max = 2 seconds
% save original_data_1000_100_2_heuristic_result  t_round_value cumulative_t_round_ratio 
% save original_data_1000_100_3_heuristic_result  tt_round_value cumulative_t_round_ratio % T_max = 3 seconds
% save original_data_1000_100_5_heuristic_result  tt_round_value cumulative_t_round_ratio %  T_max = 6 seconds
% save original_data_1000_100_7_heuristic_result  tt_round_value cumulative_t_round_ratio  %  T_max = 7 seconds
% save original_data_1000_100_9_heuristic_result  tt_round_value cumulative_t_round_ratio  %  T_max = 9 seconds

