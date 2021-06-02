function [  thisauc ] = CosPlus( train, test )
%% ����Cos+ָ�겢����AUCֵ
    D = sparse(eye(size(train,1)));                        
    % ����ϡ��ĵ�λ����
    D(logical(D)) = sum(train,2);  
    % ���ɶȾ��� ���Խ���Ԫ��Ϊͬ�±�ڵ�Ķȣ�
    pinvL = sparse(pinv( full(D - train) ));      clear D;
    % ������˹�����α��  
    Lxx = diag(pinvL);   
    % ȡ�Խ���Ԫ��
    tmp = Lxx*Lxx';
    tmp(tmp<0)=0;
    sim = pinvL ./ (tmp).^0.5; clear tmp;                         
    % �����ƶȾ���
    sim(isnan(sim)) = 0; sim(isinf(sim)) = 0;
    thisauc = CalcAUC(train,test,sim, 10000);      
    % ���⣬�����ָ���Ӧ��AUC
end
