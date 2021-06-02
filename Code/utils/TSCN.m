function [ thisauc ] = TSCN( train, test, lambda )
%% ����TSCNָ�겢����AUCֵ
    sim = train * train;     
    disp('start inv');
    disp(datestr(now, 'mm-dd HH:MM:SS'));
    % ���㹲ͬ�ھ����ƶȾ���
    I = sparse(eye(size(train,1)));
    sim = inv(I - lambda*sim) * sim;
    disp('end inv');
    disp(datestr(now, 'mm-dd HH:MM:SS'));
    % ���ƶ�ת��
    thisauc = CalcAUC(train,test,sim, 10000);     
    % ���⣬�����ָ���Ӧ��AUC
end
