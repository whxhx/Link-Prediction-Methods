function [ thisauc ] = TSRWR( train, test, lambda )
%����Random walk with restart�㷨��������ƶȾ�����������ת��
    % ����RWR���ƶȾ���
    train = train + train';
    deg = repmat(sum(train,2),[1,size(train,2)]);
    train = train ./ deg; clear deg;                                % ��ת�ƾ���
    I = sparse(eye(size(train,1)));                                 % ���ɵ�λ����
    sim = (1 - lambda) * inv(I- lambda * train') * I;
    sim = sim+sim';                                                 % ���ƶȾ���������
    train = spones(train);
    % ����������ת�ƾ���
    sim = inv(I - lambda*sim) * sim;
    sim = triu(sim,1);                                              % ȡ�����������Ծ��󣬲���ѵ�����д��ڱߵĶ�ӦԪ����Ϊ0
    sim = sim - sim.*train;                                         % ֻ�������Լ��Ͳ����ڼ����еıߵ����ƶȣ��Ի����⣩
    thisauc = CalcAUC(train,test,sim);                              % ���⣬�����ָ���Ӧ��AUC
end
