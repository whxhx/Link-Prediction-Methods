function [ thisauc ] = HDI( train, test, nodedegree)
%% ����HDIָ�겢����AUCֵ
% changed by geyao
% %     sim = train * train;    
%     sim = trainsquare;
%     % ��ɷ��ӵļ��㣬����ͬ��ͬ�ھ��㷨
%     deg_row = repmat(sum(train,1), [size(train,1),1]);
%     deg_row = deg_row .* spones(sim);
%     deg_row = max(deg_row, deg_row');  
%     % ��ɷ�ĸ�ļ��㣬����Ԫ��(i,j)��ʾȡ�˽ڵ�i�ͽڵ�j�Ķȵ����ֵ
%     sim = sim ./ deg_row; clear deg_row;     
%     % ������ƶȾ���ļ���
%     sim(isnan(sim)) = 0; sim(isinf(sim)) = 0;
%     thisauc = CalcAUC(train,test,sim, 10000);   
%     % ���⣬�����ָ���Ӧ��AUC
%     disp(thisauc);
% changed over
%    nodedegree = sum(train, 2)';
     test = triu(test);
    test_num = nnz(test);
    non_num = test_num;
    %���test��ÿ����Ե����ƶ�
    test_data = zeros(1, test_num);
    [i,j] = find(test);
    for k = 1 : length(i)
        test_data(1,k) =  sum(train(i(k),:) .* train(j(k),:)) / max(nodedegree(1, i(k)) , nodedegree(1, j(k)));
    end
    non_data = zeros(1, non_num);
    %���ѡ�񲻴��ڱ߼����еĵ��
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
                non_data(1,k) = sum(train(i,:) .* train(j,:)) / max(nodedegree(1, i) , nodedegree(1, j));
                k = k + 1;
            end
        end
    end
    labels = [ones(1,size(test_data,2)), zeros(1,size(test_data,2))];
    scores = [test_data, non_data];
    [X,Y,T,auc] = perfcurve(labels, scores, 1);
    thisauc = auc;  
end
