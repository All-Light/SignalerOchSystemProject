close all;
s = tf('s');

R = 1;
R2 = 1;
R3 = 1;
C = 1;
R4 = 1;
R5 = 1;

G = R5/R4;
K = G/(R*C);

H3 = s^2/(s^2 - R2*K*s/R3 - K)
H2 = (K/s) * H3
H1 = (K/s) * H2

figure;

subplot(3,1,1);
pzplot(H1);
legend('H1');

subplot(3,1,2)
pzplot(H2);
legend('H2');

subplot(3,1,3)
pzplot(H3);
legend('H3');

figure;
subplot(3,1,1)
impulse(H1);
legend('H1');
xlabel('Time [s]'); 
ylabel('Amplitude');
grid on

subplot(3,1,2)
impulse(H2);
legend('H2');
xlabel('Time [s]'); 
ylabel('Amplitude');
grid on

subplot(3,1,3)
impulse(H3);
legend('H3');
xlabel('Time [s]'); 
ylabel('Amplitude');
grid on


figure;
subplot(1,3,1)
bode(H1);
legend('H1');
grid on

subplot(1,3,2)
bode(H2);
legend('H2');
grid on

subplot(1,3,3)
bode(H3);
legend('H3');
grid on
