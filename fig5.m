% to plot and compare the beampattern of the UPA
% 0 in the polar figure is the direction of the x axis
% final modified in 11/10
clear
close all
xth = 12;% choose the xth beamformer
N_a = 16;
N_x = 4; N_y = 4;
c = 3e8;
fc = 3.2e9;
lambda = c/fc;
b_a = lambda/2;
c_a = [0;0;0];c_b = [-100;150;200];c_e = [-90;150;160];
d_b = norm(c_a-c_b); d_e = norm(c_a-c_e);
varphi_b = -atan(abs(c_b(1))/abs(c_b(2)));
theta_b = pi/2-atan(abs(c_b(3))/sqrt(c_b(1)^2+c_b(2)^2)); 
varphi_e = -atan(abs(c_e(1))/abs(c_e(2)));
theta_e = pi/2-atan(abs(c_e(3))/sqrt(c_e(1)^2+c_e(2)^2)); 
theta = theta_b;
phi = varphi_b;
% phi = 60/180*pi;
x = (pi/2-theta)*180/pi;
beta_0_dB = -70;% in dB
beta_0 = 10^(beta_0_dB/10);
eta_b = 3.2; eta_e = 3.2;
snrb = beta_0*d_b^(-1*eta_b);
snre = beta_0*d_e^(-1*eta_e);
delta_ = 1e-6;
P_a = 10.^((-10:2:10)./10)./(beta_0*d_b^(-1*eta_b)).*delta_^2;

load('./output/validate/5/f_1.mat')
f = reshape(double(f(xth,:).'),[N_x, N_y]);
array1 = phased.URA([N_x, N_y],lambda/2,'ArrayNormal','z','Taper',f);
load('./data/5/H_bk.mat')
load('./data/5/H_ek.mat')
H_bk = H_bk(100001:200000,:,:);
H_ek = H_ek(100001:200000,:,:);
H_bks = squeeze(H_bk(xth,:,:));
H_eks = squeeze(H_ek(xth,:,:));
[V, D] = eig(eye(N_x*N_y)+P_a(1).*H_bks'*H_bks,eye(N_x*N_y)+P_a(1).*H_eks'*H_eks);
% [V, D] = eig(eye(N_x*N_y)+P_a(1)*snrb/delta_^2.*H_bks'*H_bks,eye(N_x*N_y)+P_a(1)*snrb/delta_^2.*H_eks'*H_eks);
f_c = V(:,1)/norm(V(:,1));
f_c1 = sqrt(P_a(1)).*f_c;
array2 = phased.URA([N_x, N_y],lambda/2,'ArrayNormal','z','Taper',f_c1);


aa = pattern(array1,fc,phi*180/pi,-90:90,'Type','efield','Normalize',false);
bb = pattern(array2,fc,phi*180/pi,-90:90,'Type','efield','Normalize',false);
figure
polarplot((-90:1:90)/180*pi,aa,'LineWidth',1),hold on
polarplot((-90:1:90)/180*pi,bb,'LineWidth',1)
title('Elevation Cut (azimuth angle = -33.7°)','FontName','TimesNewRoman','FontSize',12)
legend('3D DL','3D GEVD','FontName','TimesNewRoman','FontSize',12,'Location','Best')
pax = gca;
pax.FontSize = 12;pax.FontName = 'TimesNewRoman';


