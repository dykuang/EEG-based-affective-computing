function [] = make_plots_DREAMER(matpath, version)
% '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_V_V0_False.mat'
%
    addpath('/home/dykuang/Codes/SEED/');
    locations =load('/mnt/HDD/Datasets/SEED/channel_loc.mat').locations;
    Data = load(matpath);
    ch_list = {'AF3','F7','F3','FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'};
    %set(0,'DefaultFigureVisible','off')
%     fn=fieldnames(Data);
    %loop through the fields
%     for i=1: numel(fn)
%         data_nodes = Data.(fn{i});
    i=12; % subject
        attr = strcat('S', num2str(i));
        subject_data = Data.(attr);
        
        node_data = double(subject_data(:,1:14));
        %node_data = node_data./max(abs(node_data),[],[1,2]); %renormalization
        node_mean = mean(node_data, 1);  % stats wrt time
        node_std = std(node_data, 1);
        
        region_data = double(subject_data(:,15:end-1));
        %region_data = region_data./max(abs(region_data),[],[1,2]);

        region_mean = mean(region_data, 1); % stats wrt time
        region_std = std(region_data, 1);

        region_mean_padded = zeros(1,14);
        region_std_padded = zeros(1,14);
        
        %% graph G0
        if version == 0
            for j = 1:4
                region_mean_padded(1,j) = region_mean(1,1);
                region_std_padded(1,j) = region_std(1,1);
            end
            for j = 5:7
                region_mean_padded(1,j) = region_mean(1,2);
                region_std_padded(1,j) = region_std(1,2);
            end
            for j = 8:10
                region_mean_padded(1,j) = region_mean(1,3);
                region_std_padded(1,j) = region_std(1,3);
            end
            for j = 11:14
                region_mean_padded(1,j) = region_mean(1,4);
                region_std_padded(1,j) = region_std(1,4);
            end
        %% graph G1
        elseif version == 1
            for j = 1:3
                region_mean_padded(1,j) = region_mean(1,1);
                region_std_padded(1,j) = region_std(1,1);
            end
            for j = 4:6
                region_mean_padded(1,j) = region_mean(1,2);
                region_std_padded(1,j) = region_std(1,2);
            end
            for j = 7:8
                region_mean_padded(1,j) = region_mean(1,3);
                region_std_padded(1,j) = region_std(1,3);
            end
            for j = 9:11
                region_mean_padded(1,j) = region_mean(1,4);
                region_std_padded(1,j) = region_std(1,4);
            end
            for j = 12:14
                region_mean_padded(1,j) = region_mean(1,5);
                region_std_padded(1,j) = region_std(1,5);
            end
        %% graph G2
        else
            for j = 1:5
                region_mean_padded(1,j) = region_mean(1,1);
                region_std_padded(1,j) = region_std(1,1);
            end
            for j = 6:9
                region_mean_padded(1,j) = region_mean(1,2);
                region_std_padded(1,j) = region_std(1,2);
            end
            for j = 10:14
                region_mean_padded(1,j) = region_mean(1,3);
                region_std_padded(1,j) = region_std(1,3);
            end
        end

       % renormalization
       
       node_max = max(node_mean); % renormalize to [-1,1]
       node_min = min(node_mean);
       node_std_max = max(node_std);
       node_std_min = min(node_std);

       region_max = max(region_mean);
       region_min = min(region_mean);
       region_std_max = max(region_std);
       region_std_min = min(region_std);

       node_re = 2.0 * (node_mean - node_min)./(node_max - node_min) - 1.0;
       region_re = 2.0 * (region_mean_padded - region_min)./(region_max - region_min) - 1.0;

       node_std_re = (node_std - node_std_min)./(node_std_max - node_std_min);
       region_std_re = (region_std_padded - region_std_min)./(region_std_max - region_std_min);

       figure
       subplot(1,2,1)
       [A,B,C] = plot_topography(ch_list,  node_re, true, locations, true, false, 500, 'bwr','default');  % use log map? log(1+mean)
       %colorbar off;

       subplot(1,2,2)
       [A,B,C] = plot_topography(ch_list,  region_re, true, locations, true, false, 500, 'bwr','default');
    
       figure
       subplot(1,2,1)
       [A,B,C] = plot_topography(ch_list,  node_std_re, true, locations, true, false, 500, 'hot','default'); % use log map? log(1+std)
       %colorbar off
    
       subplot(1,2,2)
       [A,B,C] = plot_topography(ch_list,  region_std_re, true, locations, true, false, 500, 'hot', 'default');
 

       %saveas(gcf,strcat(matpath(1:end-4), attr, '.png'));


%     end
    %set(0,'DefaultFigureVisible','on')
%end