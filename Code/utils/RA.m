function [ thisauc ] = RA( train, test, nodedegree )   
%% 计算RA指标并返回AUC值
% changed by geyao
%     train1 = train ./ repmat(sum(train,2),[1,size(train,1)]); 
%     % 计算每个节点的权重，1/k_i,网络规模过大时需要分块处理
%     train1(isnan(train1)) = 0; 
%     train1(isinf(train1)) = 0;
%     sim = train * train1;  clear train1; 
%     % 实现相似度矩阵的计算
%     [thisauc] = CalcAUC(train,test,sim, 10000);   
%     disp(thisauc);
%     % 评测，计算该指标对应的AUC
% changed over
%     nodedegree = sum(train, 2)';
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
            test_data(1,k) = test_data(1,k) + 1 / nodedegree(1, y(l));
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
                    non_data(1,k) = non_data(1,k) + 1 / nodedegree(1, y(l));
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
