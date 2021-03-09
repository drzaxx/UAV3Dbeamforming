% 2020/11/17 923 revised by Runze Dong
% This .m file is used to generate the CSI data
clear
N = 200000; % number of samples, 100000 for train and test respectively
N_x = 4; N_y = 4;
N_b = 6; N_e = 6;
c = 3e8;
fc = 3.2e9;
lambda_c = c/fc;
b_b = 0.05;b_e = 0.05;
b_a = lambda_c/2;% antenna spacing
c_a = [0;0;0];c_b = [-100;150;200];c_e = [-90;150;160];
rho = 0.95;
c__e = (c_e - sqrt(1-rho)*randn(3,1))/sqrt(rho);% known is c__e, real is c_e
save('.\c_e\c_e1.mat', 'c__e')% c_e for trainning
load('.\c_e\c_e1.mat')
K_b = 10^0; K_e = 10^0;
varphi_b = -atan(abs(c_b(1))/abs(c_b(2)));theta_b = pi/2-atan(abs(c_b(3))/sqrt(c_b(1)^2+c_b(2)^2)); 
phi_b = atan(abs(c_b(1))/abs(c_b(2)));vartheta_b = theta_b;
varphi_e = -atan(abs(c__e(1))/abs(c__e(2)));theta_e = pi/2-atan(abs(c__e(3))/sqrt(c__e(1)^2+c__e(2)^2)); 
phi_e = atan(abs(c__e(1))/abs(c__e(2)));vartheta_e = theta_e;
beta_0_dB = -70;% in dB
beta_0 = 10^(beta_0_dB/10);
d_b = norm(c_a-c_b); d_e = norm(c_a-c__e); 
eta_b = 3.2; eta_e = 3.2;
delta_b = sqrt(1e-12); delta_e = sqrt(1e-12);
h_bLA = exp(-1i*2*pi/lambda_c*b_b*cos(phi_b)*sin(vartheta_b).*(0:N_b-1)).';
h_eLA = exp(-1i*2*pi/lambda_c*b_e*cos(phi_e)*sin(vartheta_e).*(0:N_e-1)).';
H_temp1 = repmat((0:N_x-1)',1,N_y);
H_temp2 = repmat((0:N_y-1)',1,N_x)';
H_bLD = exp(-1i*2*pi/lambda_c*b_a*sin(theta_b).*(cos(varphi_b).*H_temp1 - sin(varphi_b).*H_temp2));
H_eLD = exp(-1i*2*pi/lambda_c*b_a*sin(theta_e).*(cos(varphi_e).*H_temp1 - sin(varphi_e).*H_temp2));
h_bLD = reshape(H_bLD,N_x*N_y,1).';
h_eLD = reshape(H_eLD,N_x*N_y,1).';
H_bL = h_bLA*h_bLD;
H_eL = h_eLA*h_eLD;
alpha_b = 0.95;alpha_e = 0.3;

H_bk = zeros(N,N_b,N_x*N_y);H_ek = zeros(N,N_b,N_x*N_y);
H_bt = zeros(N,N_b,N_x*N_y);H_et = zeros(N,N_b,N_x*N_y);
for n = 1:N
    h_H_bN = Ray_channel(N_b,N_x*N_y);
    d_H_bN = Ray_channel(N_b,N_x*N_y);
    H_bN = sqrt(alpha_b)*h_H_bN+sqrt(1-alpha_b)*d_H_bN;
    h_H_eN = Ray_channel(N_e,N_x*N_y);
    d_H_eN = Ray_channel(N_e,N_x*N_y);
    H_eN = sqrt(alpha_e)*h_H_eN+sqrt(1-alpha_e)*d_H_eN;
    H_bk(n,:,:) = sqrt(K_b/(1+K_b))*H_bL + sqrt(1/(1+K_b))*h_H_bN;
    H_ek(n,:,:) = sqrt(K_e/(1+K_e))*H_eL + sqrt(1/(1+K_e))*h_H_eN;
    H_bt(n,:,:) = sqrt(K_b/(1+K_b))*H_bL + sqrt(1/(1+K_b))*H_bN;
    H_et(n,:,:) = sqrt(K_e/(1+K_e))*H_eL + sqrt(1/(1+K_e))*H_eN;
end
    
m = num2str(1);
save(strcat('.\data\',m,'\H_bt.mat'), 'H_bt')
save(strcat('.\data\',m,'\H_et.mat'), 'H_et')
save(strcat('.\data\',m,'\H_bk.mat'), 'H_bk')
save(strcat('.\data\',m,'\H_ek.mat'), 'H_ek')

