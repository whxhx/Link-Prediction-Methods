function [  thisauc ] = RWR( train, test, lambda )
%% ����RWRָ�겢����AUCֵ
    deg = repmat(sum(train,2),[1,size(train,2)]);
    train = max(train ./ deg,0); 	clear deg;
    % ��ת�ƾ���
    I = sparse(eye(size(train,1)));                                
    % ���ɵ�λ����
    sim = (1 - lambda) * inv(I- lambda * train') * I;
    %sim = (1 - lambda) ./(I- lambda * train') * I;
    sim = sim+sim';                           
    % ���ƶȾ���������
    train = spones(train);   
    % ���ڽӾ���ԭ����Ϊ�޹����㣬���Բ����нڵ�Ķ�Ϊ0
    thisauc = CalcAUC(train,test,sim, 10000);      
    % ���⣬�����ָ���Ӧ��AUC
end
