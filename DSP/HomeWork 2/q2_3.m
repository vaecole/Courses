figure(2)
subplot(211)
hold on
plot(yn1)
plot(yn2)
legend('yn1','yn2');
hold off

xn2 = [xn zeros(1,99)];
hn2 = [hn zeros(1,99)];
Xf2 = fft(xn2);
Hf2 = fft(hn2);
Yf2 = Xf2.*Hf2;
yn3 = ifft(Yf2);
subplot(212)
plot(yn3)
legend('yn3');
title('Using DFT and IDFT with 0 Padding')