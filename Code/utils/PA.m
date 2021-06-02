function [ thisauc ] = PA( train, test, nodedegree )
%% 计算PA指标并返回AUC值
% changed by geyao
%     deg_row = sum(train,2);       
%     % 所有节点的度构成列向量，将它乘以它的转置即可
%     sim = deg_row * deg_row';  
%     clear deg_row deg_col;       
%     % 相似度矩阵计算完成
%     thisauc = CalcAUC(train,test,sim, 10000); 
%     % 评测，计算该指标对应的AUC
%     disp(thisauc);
% changed over
     
    %计算每个节点的度
%    nodedegree = sum(train, 2)';
    test = triu(test);
    test_num = nnz(test);
    non_num = test_num;
    %获得test中每个点对的相似度
    test_data = zeros(1, test_num);
    [i,j] = find(test);
    for k = 1 : length(i)
        test_data(1,k) = nodedegree(1, i(k)) * nodedegree(1, j(k));
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
                non_data(1,k) = nodedegree(1,i) * nodedegree(1, j);
                k = k + 1;
            end
        end
    end
    labels = [ones(1,size(test_data,2)), zeros(1,size(test_data,2))];
    scores = [test_data, non_data];
    [X,Y,T,auc] = perfcurve(labels, scores, 1);
    thisauc = auc;
end
