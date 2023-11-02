function Fuzzy_MatData =my_fuzzy(X)

fprintf('Finding mean, min and max matrices for the  data\n'); 

rr = (mean(X'))';
aa = (min(X'))';
bb = (max(X'))';

no_feature = size(X,1); 
Fuzzy_MatData = zeros(size(X));

fprintf('Finding the membership grade for given data set\n'); 
for i = 1:no_feature
%     Fuzzy_MatData(i, :) = pimf(X(i,:), [aa(i,1),rr(i,1),rr(i,1),bb(i,1)]);
%       Fuzzy_MatData(i, :) = trapmf(X(i,:), [aa(i,1),rr(i,1), rr(i,1),bb(i,1)]);
    Fuzzy_MatData(i, :) = smf(X(i,:), [aa(i,1), bb(i,1)]);
      
end