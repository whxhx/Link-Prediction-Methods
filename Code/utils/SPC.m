function [  thisauc, dA,wdA ] = SPC( train, test,k )
%% ����ACTָ�겢����AUCֵ
    D = sparse(eye(size(train,1)));   
    % ����ϡ��ĵ�λ����
    D(logical(D)) = sum(train,2);   
    % ���ɶȾ��󣨶Խ���Ԫ��Ϊͬ�±�ڵ�Ķȣ�
%     pinvL = sparse(pinv( full(D - train) ));   clear D;
%     % ������˹�����α��
%     Lxx = diag(pinvL);     
%     % ȡ�Խ���Ԫ��
%     Lxx = repmat(Lxx, [1,size(train,1)]);   
%     % ���Խ���Ԫ��������չΪn��n�׾���
    if nargin < 3 
        k=10;
    end
    invD = 1./D;
    invD(isinf(invD)) = 0;
    invD(isnan(invD)) = 0;
    L = full(eye(size(train,1)) - invD*train);
    [U,S] = eig(L);
    S=diag(S);
    
    %k=1000;
    [ S,I]=sort(S,'ascend');
    U=U(:,I(1:k));
    %U = U(:,S'<0.01);
    
    %U(:,1:end-k)=[];
    xii = sum(bsxfun(@times,U,U),2);
    sim = bsxfun(@plus,xii,xii')-2*U*U';
    
    %sim = pinvL;               
    sim(isnan(sim)) = 0; 
    sim(isinf(sim)) = 0;
    %sim = 1./(1+sim);
    sim = -sim;
%     sim = sim - 10*eye(size(sim,1));
    
    %sim(isinf(sim)) = 0;    %convert diagnal infs to 0, no influence
    [thisauc,dA,wdA] = CalcAUC(train,test,sim, 10000);    
    % ���⣬�����ָ���Ӧ��AUC
end 
