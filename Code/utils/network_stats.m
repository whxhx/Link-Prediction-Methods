dataname = strvcat('USAir','NS','PB','Yeast','Celegans','Power','Router'); data_num = 7;
%dataname = strvcat('Power','Router'); data_num = 2;
%dataname = strvcat('USAir'); data_num = 1;
datapath = strcat(pwd,'/data/');       %���ݼ����ڵ�·��
stats = zeros(data_num+1,5);
for ith_data = 1:data_num+1              %consider the meta data +1                           
    % ����ÿһ������
    tempcont = strcat('���ڴ���� ', int2str(ith_data), '������...');
    disp(tempcont);
    tic;
    if ith_data <= data_num
        thisdatapath = strcat(datapath,dataname(ith_data,:),'.txt');    % ��ith�����ݵ�·��
        linklist = load(thisdatapath);                                  % �������ݣ��ߵ�list��
        net = FormNet(linklist); clear linklist;                       % ���ݱߵ�list�����ڽӾ���
    else
        load(strcat(datapath,'EcoliModel.mat'));
        S = iAF1260.S;
        net = S*S';   %Adjacency matrix
        net = net - diag(diag(net));
        net = logical(net);
    end
    [largest,components] = largestcomponent(net);
    
    net1 = triu(net);
    N = size(net1,1);    %nodes number
    M = nnz(net1);     %links number
    average_degree = nnz(net)/N;
    stats(ith_data,:) = [N,M,largest,components,average_degree];    %nodes number, links number, ..., ..., 
    
    subplot(4,2,ith_data);
    vet = sum(net,2);
    histogram(vet);
    
   
    
end
