close all;
s = tf('s');


G = 1;
R = 1;
R2 = 1;
R3 = 1;
C = 1;

H1 = (G/(R*C*s))^2
H2 = G/(R*C*s)
H3 = 1 + (R2/R3)*H2 + H1

figure('Name', 'Plots', 'Position',  [50 50 1400 900]);

subplot(3,3,1);
pzplot(H1);
legend('H1');

subplot(3,3,2)
pzplot(H2);
legend('H2');

subplot(3,3,3)
pzplot(H3);
legend('H3');

subplot(3,3,4)
impulse(H1);
legend('H1');
xlabel('Time [s]'); 
ylabel('Amplitude');
grid on

subplot(3,3,5)
impulse(H2);
legend('H2');
xlabel('Time [s]'); 
ylabel('Amplitude');
grid on

subplot(3,3,6)
impulse(H3);
legend('H3');
xlabel('Time [s]'); 
ylabel('Amplitude');
grid on



subplot(3,3,7)
bode(H1);
legend('H1');
grid on

subplot(3,3,8)
bode(H2);
legend('H2');
grid on

subplot(3,3,9)
bode(H3);
legend('H3');
grid on
