function Y=index_lookup_map(X)
    Y=containers.Map();
    for i=1:length(X)
        s=num2str(X(i,:));
        Y(s)=i;
    end
