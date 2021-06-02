function [  thisauc] = Katz( train, test, lambda )
%% ����katzָ�겢����AUCֵ
    sim = inv( sparse(eye(size(train,1))) - lambda * train);   
    % �����Ծ���ļ���
    sim = sim - sparse(eye(size(train,1)));
    [thisauc] = CalcAUC(train,test,sim, 10000);   
    % ���⣬�����ָ���Ӧ��AUC
end
