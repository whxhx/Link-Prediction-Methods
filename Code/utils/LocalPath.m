function [  thisauc ] = LocalPath( train, test, lambda )
%% ����LPָ�겢����AUCֵ
    sim = train*train;    
    % ����·��
    sim = sim + lambda * (train*train*train);   
    % ����·�� + ����������·��
    thisauc = CalcAUC(train,test,sim, 10000);  
    % ���⣬�����ָ���Ӧ��AUC
end
