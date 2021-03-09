clear
SNR = -10:2:10;
load('./output/R_s/R_s_py1.mat')
plot(SNR,R_s_py,'-o','color',[0,0.4470,0.7410],'markeredgecolor',[0,0.4470,0.7410],'LineWidth',2,'MarkerSize',8);hold on;grid on
load('./output/multiEve/R_s_py1.mat')
plot(SNR,R_s_py,'->','color',[0.8500,0.3250,0.0980],'markeredgecolor',[0.8500,0.3250,0.0980],'LineWidth',2,'MarkerSize',8);
legend('Single Eve','Multiple Eves','FontSize',12,'Location','best')
xlabel('relative SNR (dB)','FontSize',12)
ylabel('average ASR (bps/Hz)','FontSize',12)
set(gca, 'XTick',-10:4:10,'FontSize',12,'FontName','TimesNewRoman')
set(gca, 'YTick',0.8:0.3:2.6,'FontSize',12,'FontName','TimesNewRoman')