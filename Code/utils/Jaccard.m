function [ thisauc ] = Jaccard( train, test, nodedegree )
%% 计算jaccard指标并返回AUC值
% changed by geyao
%     sim = train * train;   
%     sim = trainsquare;
%     % 完成分子的计算，分子同共同邻居算法
%     deg_row = repmat(sum(train,1), [size(train,1),1]);
%     deg_row = deg_row .* spones(sim);                               
%     % 只需保留分子不为0对应的元素
%     deg_row = triu(deg_row) + triu(deg_row');                      
%     % 计算节点对(x,y)的两节点的度之和
%     sim = sim./(deg_row.*spones(sim)-sim); clear deg_row;           
%     % 计算相似度矩阵 节点x与y并集的元素数目 = x与y的度之和 - 交集的元素数目
%     sim(isnan(sim)) = 0; sim(isinf(sim)) = 0;
%     thisauc = CalcAUC(train,test,sim, 10000);      
%     disp(thisauc);
%     % 评测，计算该指标对应的AUC
% changed over
%      nodedegree = sum(train, 2)';
    test = triu(test);
    test_num = nnz(test);
    non_num = test_num;
    %获得test中每个点对的相似度
    test_data = zeros(1, test_num);
    [i,j] = find(test);
    for k = 1 : length(i)
        test_data(1,k) = sum(train(i(k),:) .* train(j(k),:)) / (nodedegree(1, i(k)) + nodedegree(1, j(k)) - sum(train(i(k),:) .* train(j(k),:)));
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
                non_data(1,k) = sum(train(i,:) .* train(j,:)) / (nodedegree(1, i) + nodedegree(1, j) - sum(train(i,:) .* train(j,:)));
                k = k + 1;
            end
        end
    end
    labels = [ones(1,size(test_data,2)), zeros(1,size(test_data,2))];
    scores = [test_data, non_data];
    [X,Y,T,auc] = perfcurve(labels, scores, 1);
    thisauc = auc;
end
