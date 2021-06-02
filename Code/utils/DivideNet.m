function [ train, test ] = DivideNet( net, ratioTrain)      %problem found, double select test link because of non-triangular matrix
%%����ѵ�����Ͳ��Լ�����֤ѵ������ͨ
    net = triu(net) - diag(diag(net));  % convert to upper triangular matrix
    num_testlinks = ceil((1-ratioTrain) * nnz(net));      
    % ȷ�����Լ��ı���Ŀ
    [xindex, yindex] = find(net);  linklist = [xindex yindex];    
    % �����磨�ڽӾ��������еı��ҳ���������linklist  
    clear xindex yindex;  
    % Ϊÿ�������ñ�־λ���ж��Ƿ���ɾ��
%     test = sparse(size(net,1),size(net,2));   
    links = zeros(num_testlinks, 2);
    i = 1;
    while (i <= num_testlinks)               %For power dataset, maximum 636 test links <660 expected. 
        if length(linklist) <= 2
            break;
        end
        %---- ���ѡ��һ����
        index_link = ceil(rand(1) * length(linklist));
        
        uid1 = linklist(index_link,1); 
        uid2 = linklist(index_link,2);    
        net(uid1,uid2) = 0;  
        links(i, 1) = uid1;
        links(i, 2) = uid2;
        i = i + 1;
        
          %% 
%         %---- �ж���ѡ�����˽ڵ�uid1��uid2�Ƿ�ɴ���ɴ���ɷ�����Լ�������������ѡһ����
%          
%         % �������ߴ���������ȥ�����ж��ڵ���������Ƿ���ͨ
%         tempvector = net(uid1,:);
%         % ȡ��uid1һ���ɴ�ĵ㣬������һά����
%         sign = 0;  
%         % ��Ǵ˱��Ƿ���Ա��Ƴ���sign=0��ʾ���ɣ� sign=1��ʾ����
%         %changed by geyao
% %         uid1TOuid2 = tempvector * net + tempvector;        
%         % changed over
%         [xindex, yindex] = find(tempvector);
%         uid1TOuid2 = tempvector;
%         for i = 1 : length(yindex)
%             uid1TOuid2 = uid1TOuid2 + net(yindex(i),:);
%         end
%         % uid1TOuid2��ʾ�����ڿɴ�ĵ�
%         if uid1TOuid2(uid2) > 0
%             sign = 1;               
%             % �������ɴ�
%         else
            % changed by geyao
%             while (nnz(spones(uid1TOuid2) - tempvector) ~=0)   
%             % ֱ���ɴ�ĵ㵽���ȶ�״̬����Ȼ���ܵ���uid2���˱߾Ͳ��ܱ�ɾ��
%                 tempvector = spones(uid1TOuid2);
%                 uid1TOuid2 = tempvector * net + tempvector;    
%                 % �˲���uid1TOuid2��ʾK���ڿɴ�ĵ�
%                 if uid1TOuid2(uid2) > 0
%                     sign = 1;      
%                      % ĳ���ڿɴ�
%                     break;
%                 end
%             end
           % changed over
%             while (nnz(spones(uid1TOuid2) - tempvector) ~=0)   
%             % ֱ���ɴ�ĵ㵽���ȶ�״̬����Ȼ���ܵ���uid2���˱߾Ͳ��ܱ�ɾ��
%                 tempvector = spones(uid1TOuid2);
%                 [xindex, yindex] = find(tempvector);
%                 uid1TOuid2 = tempvector;
%                 for i = 1 : length(yindex)
%                     uid1TOuid2 = uid1TOuid2 + net(yindex(i),:);
%                 end
%                 % �˲���uid1TOuid2��ʾK���ڿɴ�ĵ�
%                 if uid1TOuid2(uid2) > 0
%                     sign = 1;      
%                      % ĳ���ڿɴ�
%                     break;
%                 end
%             end
%         end 
        % ����-�ж�uid1�Ƿ�ɴ�uid2
        
%        sign = 1;  % changed... keep all selected links in test, no matter whether connect

        %% 
        
        %----���˱߿�ɾ������֮������Լ��У������˱ߴ�linklist���Ƴ�
%         if sign == 1 %�˱߿���ɾ��
%             linklist(index_link,:) = []; 
%             test(uid1,uid2) = 1;
%         else
%             linklist(index_link,:) = [];
%             net(uid1,uid2) = 1;   
%         end   
        % ����-�жϴ˱��Ƿ����ɾ��������Ӧ����
    end   
    % ������while��-���Լ��еı�ѡȡ���
    test = sparse(links(:,1), links(:,2), 1, size(net,1), size(net,2));
    train = net + net';  test = test + test';
    % ����Ϊѵ�����Ͳ��Լ�
end
