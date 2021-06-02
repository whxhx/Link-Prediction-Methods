function [  thisauc ] = ACT( train, test )
%% ����ACTָ�겢����AUCֵ
    D = sparse(eye(size(train,1)));   
    % ����ϡ��ĵ�λ����
    D(logical(D)) = sum(train,2);   
    % ���ɶȾ��󣨶Խ���Ԫ��Ϊͬ�±�ڵ�Ķȣ�
    pinvL = sparse(pinv( full(D - train) ));   clear D;
    % ������˹�����α��
    Lxx = diag(pinvL);     
    % ȡ�Խ���Ԫ��
    Lxx = repmat(Lxx, [1,size(train,1)]);   
    % ���Խ���Ԫ��������չΪn��n�׾���
    
    sim = 1./(1+Lxx + Lxx' - 2*pinvL);               
    sim(isnan(sim)) = 0; 
    %sim = -(Lxx + Lxx' - 2*pinvL);       %changed
    
    %sim(isinf(sim)) = 0;    %convert diagnal infs to 0, no influence
    thisauc = CalcAUC(train,test,sim, 10000);    
    % ���⣬�����ָ���Ӧ��AUC
end 
