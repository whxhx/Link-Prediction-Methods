function [ thisauc ] = MFI( train,test )
%% ����MFIָ�겢����AUCֵ
    I = sparse(eye(size(train,1)));                                 
    % ���ɵ�λ����
    D = I;
    D(logical(D)) = sum(train,2);         
    % ���ɶȾ��󣨶Խ���Ԫ��Ϊͬ�±�ڵ�Ķȣ�
    L = D - train;          
    clear D;              
    % ������˹����
    sim = inv(I + L);      
    clear I L;       
    % ���ƶȾ���ļ���
    thisauc = CalcAUC(train,test,sim, 10000);   
    % ���⣬�����ָ���Ӧ��AUC
end
