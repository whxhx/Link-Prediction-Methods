function [  thisauc ] = LHNII( train, test, lambda )   
%% ����LHN2ָ�겢����AUCֵ
    M = nnz(train)/2;
    % �����еı���
    D = sparse(eye(size(train,1)));                                 
    D(logical(D)) = sum(train,2);   
    % ���ɶȾ��� ���Խ���Ԫ��Ϊͬ�±�ڵ�Ķȣ�
    D = inv(D);  
    %D = inv(D+0.1*eye(size(D,1)));  
    % ��Ⱦ���������
    maxeig = max(eigs(train));  
    % ���ڽӾ�����������ֵ
    tempmatrix = (sparse(eye(size(train,1))) - lambda/maxeig * train); 
    tempmatrix = inv(tempmatrix);
    sim = 2 * M * maxeig * D * tempmatrix * D;   clear D tempmatrix;
    % ������ƶȾ���ļ���
    thisauc = CalcAUC(train,test,sim, 10000);    
    % ���⣬�����ָ���Ӧ��AUC
end
