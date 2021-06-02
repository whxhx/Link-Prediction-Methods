function [  thisauc ] = SRW( train, test, steps, lambda )
%% ����SRWָ�겢����AUCֵ
    deg = repmat(sum(train,2),[1,size(train,2)]);
    train = max(train ./ deg,0); clear deg;
    % ��ת�ƾ���
    I = sparse(eye(size(train,1)));                                 
    % ���ɵ�λ����
    tempsim = I;                            
    % �����ݴ�ÿ���ĵ������
    stepi = 0; sim = sparse(size(train,1),size(train,2));           
    % ������ߵĵ��� sim�����洢ÿ�������ķ�ֵ֮��
    while(stepi < steps)
        tempsim = (1-lambda)*I + lambda * train' * tempsim;
        stepi = stepi + 1;
        sim = sim + tempsim;
    end
    sim = sim+sim';                        
    % ���ƶȾ���������
    train = spones(train);   
    %���ڽӾ���ԭ����Ϊ�޹����㣬���Բ����нڵ�Ķ�Ϊ0
    thisauc = CalcAUC(train,test,sim, 10000);    
    % ���⣬�����ָ���Ӧ��AUC
end
