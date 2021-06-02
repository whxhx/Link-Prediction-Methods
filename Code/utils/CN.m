function [ thisauc ] = CN( train, test )
%% ����CNָ�겢����AUCֵ
% changed by geyao
%     sim = train * train;        
    % ���ƶȾ���ļ���
%     thisauc = CalcAUC(train,test,sim, 10000);

%     thisauc = CalcAUC(train,test,trainsquare, 10000);
% changed over
    test = triu(test);
    test_num = nnz(test);
    non_num = test_num;
%     %���test��ÿ����Ե����ƶ�
    test_data = zeros(1, test_num);
    [i,j] = find(test);
    for k = 1 : length(i)
        test_data(1,k) = train(i(k),:) * train(:,j(k));
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
                non_data(1,k) = train(i,:) * train(:,j);
                k = k + 1;
            end
        end
    end
    labels = [ones(1,size(test_data,2)), zeros(1,size(test_data,2))];
    scores = [test_data, non_data];
    [X,Y,T,auc] = perfcurve(labels, scores, 1);
    thisauc = auc;
end
