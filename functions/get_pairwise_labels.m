function [B, Labels] = get_pairwise_labels(ind1,ind2,model,Data)
c=1;
I= size(Data,2)-1;
N1 =length(ind1);
N2=length(ind2);
Data_test = zeros(N1*N2, I);
for i=ind1
    for j=ind2
        Data_test(c,:)=abs(Data(i,2:end) -Data(j,2:end));
        %B(i,j)=round(model(Data_test(c,:)'));
        c=c+1;
    end
end
 Labels = round(model(Data_test'));
 %Labels = predict(model,Data_test);
 B = reshape(Labels,[N2,N1])';
%  B= B+B';
%  B(B==2)=1;

end