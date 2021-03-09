%csi_gen_multiEve
clear
N = 200000; % number of samples, 100000 for train and test respectively
N_x = 4; N_y = 4;
N_b = 6; N_e = 6;
c = 3e8;
fc = 3.2e9;
lambda_c = c/fc;
b_b = 0.05;b_e = 0.05;
b_a = lambda_c/2;% antenna spacing
c_a = [0;0;0];c_b = [-100;150;200];c_e1 = [-130;80;160];c_e2 = [-110;200;180];
rho = 0.95;
c__e1 = (c_e1 - sqrt(1-rho)*randn(3,1))/sqrt(rho);% known is c__e, real is c_e
c__e2 = (c_e2 - sqrt(1-rho)*randn(3,1))/sqrt(rho);
save('.\c_e\c_e_multiEve2.mat', 'c__e1')% c_e for trainning
save('.\c_e\c_e_multiEve3.mat', 'c__e2')% c_e for trainning
K_b = 10^0; K_e = 10^0;

c__e = c__e1;
varphi_e = -atan(abs(c__e(1))/abs(c__e(2)));theta_e = pi/2-atan(abs(c__e(3))/sqrt(c__e(1)^2+c__e(2)^2)); 
phi_e = atan(abs(c__e(1))/abs(c__e(2)));vartheta_e = theta_e;
h_eLA = exp(-1i*2*pi/lambda_c*b_e*cos(phi_e)*sin(vartheta_e).*(0:N_e-1)).';
H_temp1 = repmat((0:N_x-1)',1,N_y);
H_temp2 = repmat((0:N_y-1)',1,N_x)';
H_eLD = exp(-1i*2*pi/lambda_c*b_a*sin(theta_e).*(cos(varphi_e).*H_temp1 - sin(varphi_e).*H_temp2));
h_eLD = reshape(H_eLD,N_x*N_y,1).';
H_eL = h_eLA*h_eLD;
alpha_b = 0.95;alpha_e = 0.3;
H_ek = zeros(N,N_b,N_x*N_y);
H_et = zeros(N,N_b,N_x*N_y);
for n = 1:N
    h_H_eN = Ray_channel(N_e,N_x*N_y);
    d_H_eN = Ray_channel(N_e,N_x*N_y);
    H_eN = sqrt(alpha_e)*h_H_eN+sqrt(1-alpha_e)*d_H_eN;
    H_ek(n,:,:) = sqrt(K_e/(1+K_e))*H_eL + sqrt(1/(1+K_e))*h_H_eN;
    H_et(n,:,:) = sqrt(K_e/(1+K_e))*H_eL + sqrt(1/(1+K_e))*H_eN;
end
save('.\data\multiEve\H_ek2.mat', 'H_ek')
save('.\data\multiEve\H_et2.mat', 'H_et')

c__e = c__e2;
varphi_e = -atan(abs(c__e(1))/abs(c__e(2)));theta_e = pi/2-atan(abs(c__e(3))/sqrt(c__e(1)^2+c__e(2)^2)); 
phi_e = atan(abs(c__e(1))/abs(c__e(2)));vartheta_e = theta_e;
h_eLA = exp(-1i*2*pi/lambda_c*b_e*cos(phi_e)*sin(vartheta_e).*(0:N_e-1)).';
H_temp1 = repmat((0:N_x-1)',1,N_y);
H_temp2 = repmat((0:N_y-1)',1,N_x)';
H_eLD = exp(-1i*2*pi/lambda_c*b_a*sin(theta_e).*(cos(varphi_e).*H_temp1 - sin(varphi_e).*H_temp2));
h_eLD = reshape(H_eLD,N_x*N_y,1).';
H_eL = h_eLA*h_eLD;
alpha_b = 0.95;alpha_e = 0.3;
H_ek = zeros(N,N_b,N_x*N_y);
H_et = zeros(N,N_b,N_x*N_y);
for n = 1:N
    h_H_eN = Ray_channel(N_e,N_x*N_y);
    d_H_eN = Ray_channel(N_e,N_x*N_y);
    H_eN = sqrt(alpha_e)*h_H_eN+sqrt(1-alpha_e)*d_H_eN;
    H_ek(n,:,:) = sqrt(K_e/(1+K_e))*H_eL + sqrt(1/(1+K_e))*h_H_eN;
    H_et(n,:,:) = sqrt(K_e/(1+K_e))*H_eL + sqrt(1/(1+K_e))*H_eN;
end
save('.\data\multiEve\H_ek3.mat', 'H_ek')
save('.\data\multiEve\H_et3.mat', 'H_et')
