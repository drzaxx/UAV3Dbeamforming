% to plot the R_s comparison fig
%%%%%%%%%%%%%%%%%%
% R_s_3D and R_s_2D are the R_s of DL, respectively
% R_s_3D1 amd R_s_2D1 are the R_s of MRT, respectively
% R_s_C3D and R_s_C2D are the R_s of conventional beamforming method under
% 3D and 2D, respectively
%%%%%%%%%%%%%%%%%%
clear
R_s_3D = zeros(1,11);R_s_3D1 = R_s_3D; R_s_C3D = R_s_3D;
R_s_2D = zeros(1,11);R_s_2D1 = R_s_2D; R_s_C2D = R_s_2D;
x = num2str(1);% data set change
N_x = 4; N_y = 4;
N_b = 6; N_e = 6;
c = 3e8;
fc = 3.2e9;
lambda = c/fc;
b_a = lambda/2;
c_a = [0;0;0];c_b = [-100;150;200];c_e = [-90;150;160];
varphi_b = -atan(abs(c_b(1))/abs(c_b(2)));
theta_b = pi/2-atan(abs(c_b(3))/sqrt(c_b(1)^2+c_b(2)^2)); 
theta = theta_b;
phi = varphi_b;
beta_0_dB = -70;% in dB
beta_0 = 10^(beta_0_dB/10);
eta_b = 3.2; eta_e = 3.2;
d_b = norm(c_a-c_b); d_e = norm(c_a-c_e);
snrb = beta_0*d_b^(-1*eta_b);
snre = beta_0*d_e^(-1*eta_e);
delta_ = 1e-6;
P_a = 10.^((-10:2:10)./10)./(beta_0*d_b^(-1*eta_b)).*delta_^2;
N = 100000;
R_sb = zeros(N,1);R_se = zeros(N,1);R_sb1 = zeros(N,1);R_se1 = zeros(N,1);
R_sb_mrt = zeros(N,1);R_se_mrt = zeros(N,1);R_sb_mrt1 = zeros(N,1);R_se_mrt1 = zeros(N,1);
R_sb_c = zeros(N,1);R_se_c = zeros(N,1);R_sb_c1 = zeros(N,1);R_se_c1 = zeros(N,1);

load(strcat('./data/',x,'/H_bt.mat'))
load(strcat('./data/',x,'/H_et.mat'))
load(strcat('./data/',x,'/H_bk.mat'))
load(strcat('./data/',x,'/H_ek.mat'))  
H_bt = H_bt(100001:200000,:,:);
H_et = H_et(100001:200000,:,:);
H_bk = H_bk(100001:200000,:,:);
H_ek = H_ek(100001:200000,:,:);
A = zeros(N_x, N_y);
for p = 1 : N_x
    for q = 1 : N_y
        A(p, q) = exp(1j*2*pi/lambda*b_a*(-(q-1)*sin(phi)*sin(theta)+(p-1)*cos(phi)*sin(theta)));
    end
end
    
for i = 1:11
    f_mrt = reshape(sqrt(P_a(i))/4.*A,[N_x*N_y,1]);
    load(strcat('./output/validate/',x,'/f_',num2str(i),'.mat'))
    load(strcat('./output/validate/',x,'/G_',num2str(i),'.mat'))
    for n = 1:N
        fs = f(n,:).';
        Gs = squeeze(G(n,:,:));
        H_bts = squeeze(H_bt(n,:,:));
        H_ets = squeeze(H_et(n,:,:));
        R_sb(n,:) = log2(det(eye(N_b)+snrb.*H_bts*fs*(H_bts*fs)'/(snrb.*H_bts*Gs*(H_bts*Gs)'+delta_^2.*eye(N_b))));
        R_se(n,:) = log2(det(eye(N_e)+snre.*H_ets*fs*(H_ets*fs)'/(snre.*H_ets*Gs*(H_ets*Gs)'+delta_^2.*eye(N_e))));
        R_sb_mrt(n,:) = log2(det(eye(N_b)+snrb/delta_^2.*H_bts*f_mrt*(H_bts*f_mrt)'));
        R_se_mrt(n,:) = log2(det(eye(N_e)+snre/delta_^2.*H_ets*f_mrt*(H_ets*f_mrt)'));
        H_bks = squeeze(H_bk(n,:,:));
        H_eks = squeeze(H_ek(n,:,:));
        [V, D] = eig(eye(N_x*N_y)+P_a(i).*H_bks'*H_bks,eye(N_x*N_y)+P_a(i).*H_eks'*H_eks);
        f_c = V(:,1)/norm(V(:,1));
        f_c1 = sqrt(P_a(i)).*f_c;
        R_sb_c(n,:) = log2(det(eye(N_b)+snrb/delta_^2.*H_bts*f_c1*(H_bts*f_c1)'));
        R_se_c(n,:) = log2(det(eye(N_e)+snre/delta_^2.*H_ets*f_c1*(H_ets*f_c1)'));
    end
    R_s_3D(i) = real(mean(R_sb) - mean(R_se));
    R_s_3D1(i) = real(mean(R_sb_mrt) - mean(R_se_mrt));
    R_s_C3D(i) = real(mean(R_sb_c) - mean(R_se_c));
end


figure(1)
SNR = -10:2:10;
plot(SNR,R_s_3D,'-o','color',[0,0.4470,0.7410],'markeredgecolor',[0,0.4470,0.7410],'LineWidth',2,'MarkerSize',8);hold on;grid on
plot(SNR,R_s_C3D,'->','color',[0.8500,0.3250,0.0980],'markeredgecolor',[0.8500,0.3250,0.0980],'LineWidth',2,'MarkerSize',8);
legend('3D DL','3D GEVD','FontSize',12,'Location','best')
set(gca, 'XTick',-10:4:10,'FontSize',12,'FontName','TimesNewRoman')
xlabel('relative SNR (dB)','FontSize',12)
ylabel('average ASR (bps/Hz)','FontSize',12)