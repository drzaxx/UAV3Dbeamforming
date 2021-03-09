function H = Ray_channel(A,B)
% Rayleigh Channel Model
%  Input : A  : Row
%  Input : B  : Column
%  Output: H  : Channel matrix
%  2020/08/29 Runze Dong


for i = 1:A
    H(i,:) = (randn(1,B)+1i*randn(1,B))/sqrt(2);
end
