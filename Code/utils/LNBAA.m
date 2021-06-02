function [ thisauc ] = LNBAA( train,test )
%% 计算局部朴素贝叶斯模型性AA指标并返回AUC值
    %changed by geyao    
%     s = size(train,1)*(size(train,1)-1) / nnz(train) -1;  
%     % 计算每个网络中的常量s
%     tri = diag(train*train*train)/2;     
%     % 计算每个点所在的三角形个数
%     tri_max = sum(train,2).*(sum(train,2)-1)/2;  
%     % 每个点最大可能所在的三角形个数
%     R_w = (tri+1)./(tri_max+1); clear tri tri_max; 
%     % 接下来几步是按照公式度量每个点的角色  
%     SR_w = (log(s)+log(R_w))./log(sum(train,2)); clear s R_w;
%     SR_w(isnan(SR_w)) = 0; SR_w(isinf(SR_w)) = 0;

%     SR_w = repmat(SR_w,[1,size(train,1)]) .* train;   
%     % 节点的角色计算完毕
%     sim = spones(train) * SR_w;   clear SR_w;                       
%     % 将节点对（x,y）的共同邻居的角色量化值相加即可
%     thisauc = CalcAUC(train,test,sim, 10000);
%     % 评测，计算该指标对应的AUC
    % changed over
    train = spones(train);
    s = size(train,1)*(size(train,1)-1) / nnz(train) -1;  
    % 计算每个网络中的常量s
    tri = zeros(1, size(train, 1));
    for i = 1 : size(train, 1)
        neighbors = train(i,:);
        [x, y] = find(neighbors);
        for j = 1 : length(y)
            for k = j : length(y)
                if j ~= k && train(y(j), y(k))
                    tri(i) = tri(i) + 1;
                end
            end
        end
    end   
    % 计算每个点所在的三角形个数
    tri_max = sum(train,2).*(sum(train,2)-1)/2;  
    % 每个点最大可能所在的三角形个数
    R_w = (2 * tri+1)./(2 * (tri_max' - tri)+1); clear tri tri_max; 
    % 接下来几步是按照公式度量每个点的角色  
    SR_w = (log(s)+log(R_w)) ./log(sum(train,2)'); clear s R_w;
    SR_w(isnan(SR_w)) = 0; SR_w(isinf(SR_w)) = 0;

    test = triu(test);
    test_num = nnz(test);
    non_num = test_num;
    %获得test中每个点对的相似度
    test_data = zeros(1, test_num);
    [i,j] = find(test);
    for k = 1 : length(i)
        cn = train(i(k),:) .* train(j(k),:);
        [x, y] = find(cn);
        for l = 1 : length(y)
            test_data(1,k) = test_data(1, k) + SR_w(1, y(l));
        end
    end
    non_data = zeros(1, non_num);
    %随机选择不存在边集合中的点对
    limiti = randperm(size(train,1),2*ceil(sqrt(non_num)));
    limitj = randperm(size(train,2),2*ceil(sqrt(non_num)));
    k = 1;
    for i = limiti
        if k > non_num
              break
        end
        for j = limitj
            if k > non_num
              break
            end
            if ~train(i,j) && ~test(i,j) && i~= j
                cn = train(i,:) .* train(j,:);
                [x, y] = find(cn);
                for l = 1 : length(y)
                    non_data(1,k) = non_data(1,k) + SR_w(1, y(l));
                end
                k = k + 1;
            end
        end
    end
    labels = [ones(1,size(test_data,2)), zeros(1,size(test_data,2))];
    scores = [test_data, non_data];
    [X,Y,T,auc] = perfcurve(labels, scores, 1);
    thisauc = auc;
end
