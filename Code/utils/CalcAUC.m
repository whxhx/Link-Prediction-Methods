function [ auc ] = CalcAUC( train, test, sim, n )    %��û�зǽ����㷨��
%% ����AUC�������������ƶȾ���
   sim = triu(sim - sim.*train) - diag(diag(sim));
    % ֻ�������Լ��Ͳ����ڱ߼����еıߵ����ƶȣ��Ի����⣩
    %changed by geyao

%  non = 1 - train - test - eye(max(size(train,1),size(train,2)));
    test = triu(test);
%     non = triu(non);
    % �ֱ�ȡ���Լ��Ͳ����ڱ߼��ϵ������Ǿ�������ȡ�����Ƕ�Ӧ�����ƶȷ�ֵ
    test_num = nnz(test);
%     non_num = nnz(non);
    non_num = test_num;
%     test_rd = ceil( test_num * rand( 1, n));  
%     % ceil��ȡ���ڵ��ڵ���С������nΪ�����ȽϵĴ���
%     non_rd = ceil( non_num * rand( 1, n));
    test_pre = sim .* test;
%     non_pre = sim .* non;
    test_data =  test_pre( test ~= 0 )';   
    non_data = zeros(1, non_num);
    limiti = randperm(size(sim,1),2*ceil(sqrt(non_num)));
    limitj = randperm(size(sim,2),2*ceil(sqrt(non_num)));
    k = 1;
    for i = limiti
        if k > non_num
              break
        end
        for j = limitj
            if k > non_num
              break
            end
            if sim(i, j) && ~train(i,j) && ~test(i,j) && i~= j
                non_data(1,k) = sim(i, j);
                k = k + 1;
            end
        end
    end
%     randnum = randperm(non_num, test_num);
%     non_data = non_pre(randnum);
     

    % ��������test ���ϴ��ڵıߵ�Ԥ��ֵ
%     non_data =  non_pre( non ~= 0 )';   
    %changed over

    % ��������nonexist���ϴ��ڵıߵ�Ԥ��ֵ
%     test_rd = test_data( test_rd );
%     non_rd = non_data( non_rd );
%     %clear test_data non_data;
%     n1 = length( find(test_rd > non_rd) );  
%     n2 = length( find(test_rd == non_rd));
%     auc = ( n1 + 0.5*n2 ) / n;
    
    % matlab�Դ�����perfcurve
    %changed by geyao
    labels = [ones(1,size(test_data,2)), zeros(1,size(test_data,2))];
%    labels = [ones(1,size(test_data,2)), zeros(1,size(non_data,2))];
    %changed over
    scores = [test_data, non_data];
    [X,Y,T,auc] = perfcurve(labels, scores, 1);

       
    %MATLAB calculate confusion matrix, ����ʵ��ʱע�͵�
%     for runthis = 1:0
%     ratio = 1;
%     labels = [ones(1,size(test_data,2)), 2*ones(1,size(non_data,2))];
%     scores = [test_data, non_data];
%     [y,i] = sort(scores,2,'descend');
%     y = y(:,1:test_num*ratio);
%     i = i(:,1:test_num*ratio);
%     g1 = labels(i);
%     g2 = ones(test_num*ratio,1);
%     C = confusionmat(g1,g2)
%     precision = C(1,1)/test_num
%     
%     figure(1);
%     plot(X,Y)
%     xlabel('False positive rate')
%     ylabel('True positive rate')
%     title('ROC');
%     end
end
% function [auc] = CalcAUC(train, test, sim, n)  
%     sim = triu(sim - sim.*train) - diag(diag(sim));
%     % ֻ�������Լ��Ͳ����ڱ߼����еıߵ����ƶȣ��Ի����⣩
%     non = 1 - train - test - eye(max(size(train,1),size(train,2)));
%     test = triu(test);
%     non = triu(non);
%     % �ֱ�ȡ���Լ��Ͳ����ڱ߼��ϵ������Ǿ�������ȡ�����Ƕ�Ӧ�����ƶȷ�ֵ
%     test_num = nnz(test);
%     non_num = nnz(non);
% %     test_rd = ceil( test_num * rand( 1, n));  
% %     % ceil��ȡ���ڵ��ڵ���С������nΪ�����ȽϵĴ���
% %     non_rd = ceil( non_num * rand( 1, n));
%     test_pre = sim .* test;
%     non_pre = sim .* non;
%     test_data =  test_pre( test ~= 0 )';  
%     % ��������test ���ϴ��ڵıߵ�Ԥ��ֵ
%     non_data =  non_pre( non ~= 0 )';   
%     % ��������nonexist���ϴ��ڵıߵ�Ԥ��ֵ
% %     test_rd = test_data( test_rd );
% %     non_rd = non_data( non_rd );
% %     %clear test_data non_data;
% %     n1 = length( find(test_rd > non_rd) );  
% %     n2 = length( find(test_rd == non_rd));
% %     auc = ( n1 + 0.5*n2 ) / n;
%     
%     % matlab�Դ�����perfcurve
%     target = [ones(1,size(test_data,2)), zeros(1,size(non_data,2))];
%     score = [test_data, non_data];
% len = length(score);                % number of patterns  
% if len ~= length(target)  
% error('The length of tow input vectors should be equal\n');  
% end  
% P = 0;    % number of Positive pattern  
% N = 0;    % number of Negative pattern  
% Lp = 1;
% Ln = 0;
% for i = 1:len  
% if target(i) == Lp  
%   P = P + 1;  
% elseif target(i) == Ln  
%   N = N + 1;  
% else  
%   error('Wrong target value');  
% end  
% end  
%   
% % sort "L"  in decending order by scores  
% score = score(:);  
% target = target(:);  
% L = [score target];  
% L = sortrows(L,1);  
% index = len:-1:1;  
% index = index';     %'  
% L = L(index,:);  
%   
% fp = 0;   fp_pre = 0;   % number of False Positive pattern  
% tp = 0;   tp_pre = 0;   % number of True Positive pattern.  
% score_pre = -10000;  
% curve = [];  
% auc = 0;  
% for i = 1:len  
% if L(i,1) ~= score_pre  
%   curve = [curve; [fp/N, tp/P, L(i,1)]];  
%   auc = auc + trapezoid(fp, fp_pre, tp, tp_pre);  
%     
%   score_pre = L(i,1);  
%     
%   fp_pre = fp;  
%   tp_pre = tp;  
% end  
%   
% if L(i,2) == Lp  
%   tp = tp + 1;  
% else  
%   fp = fp + 1;  
% end  
% end  
% curve = [curve; [1,1,0]]  
% auc = auc / P / N;  
% auc = auc + trapezoid(1, fp_pre/N, 1, tp_pre/P)  
%   
%   
% % calculat the area of trapezoid  
% function area = trapezoid(x1,x2,y1,y2)  
% a = abs(x1-x2);  
% b = abs(y1+y2);  
% area = a * b / 2;  