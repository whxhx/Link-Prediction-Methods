function [ thisauc ] = TSAA( train, test, lambda )
%����Common Neighbor�㷨��������ƶȾ�����������ת��
    % ����AA���ƶȾ���
    train = train + train';
    train1 = train ./ repmat(log(sum(train,2)),[1,size(train,1)]);  % ����ÿ���ڵ��Ȩ�أ�1/log(k_i)
                                                                    % ע�⣺�����ģ����repmat�������������ʱ��Ҫ�ֿ鴦��
    train1(isnan(train1)) = 0; train1(isinf(train1)) = 0;           % ������Ϊ0�õ����쳣ֵ��Ϊ0
    sim = train * train1;      clear train1;                        % ʵ�����ƶȾ���ļ���
    % ����������ת�ƾ���
    I = sparse(eye(size(train,1)));
    sim = inv(I - lambda*sim) * sim;
    sim = triu(sim,1);                                              % ȡ�����������Ծ��󣬲���ѵ�����д��ڱߵĶ�ӦԪ����Ϊ0
    sim = sim - sim.*train;                                         % ֻ�������Լ��Ͳ����ڼ����еıߵ����ƶȣ��Ի����⣩
    thisauc = CalcAUC(train,test,sim);                              % ���⣬�����ָ���Ӧ��AUC
end
