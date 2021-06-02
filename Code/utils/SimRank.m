function [  thisauc ] = SimRank( train, test, lambda)
%% ����SimRankָ�겢����AUCֵ
    deg = sum(train,1);     
    % ��ڵ����ȣ�������������������
    lastsim = sparse(size(train,1), size(train,2)); 
    % �洢ǰһ���ĵ����������ʼ��Ϊȫ0����
    sim = sparse(eye(size(train,1))); 
    ntrain = train.*repmat(max(1./deg,0),size(train,1),1);
    for iter = 1:5
        sim = max(lambda*(ntrain'*sim*ntrain),eye(size(train,1)));
    end
    disp(datestr(now, 'mm-dd HH:MM:SS'));
    
    
%     % �洢��ǰ���ĵ����������ʼ��Ϊ��λ����
%     while(sum(sum(abs(sim-lastsim)))>0.00001)      %original = 0.0000001
%     % ��������̬���ж�����
%         %sum(sum(abs(sim-lastsim)))
%         lastsim = sim;  sim = sparse(size(train,1), size(train,2));                                           
%         for nodex = 1:size(train,1)-1      
%         %��ÿһ�Խڵ��ֵ���и���
%             if deg(nodex) == 0
%                 continue;
%             end
%             for nodey = nodex+1:size(train,1)         
%                 
%             %-----����x���ھӺ͵�y���ھ�����ɵ����нڵ�Ե�ǰһ������������
%                 if deg(nodey) == 0
%                     continue;
%                 end
%                 sim(nodex,nodey) = lambda * sum(sum(lastsim(train(:,nodex)==1,train(:,nodey)==1))) / (deg(nodex)*deg(nodey));
%                 %nodex, nodey
%             end
%         end
%         sim = sim+sim'+ sparse(eye(size(train,1)));
%     end
    thisauc = CalcAUC(train,test,sim, 10000);    
    % ���⣬�����ָ���Ӧ��AUC
end
