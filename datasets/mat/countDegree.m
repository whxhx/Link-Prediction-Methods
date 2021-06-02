count=[];
fullJazz=full(net);
for i=1:198
    count=[count;sum(fullJazz(i,:))];
end