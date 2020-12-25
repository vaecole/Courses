hn = load('data/hn_q3.mat').hn;
xn = load('data/xn_q3.mat').xn;
yn = conv(xn, hn);
fs = 100;
fx = (1:fs/length(xn):fs/2);
fy = (1:fs/length(yn):fs/2);
Xf = fft(xn.*hann(length(xn))');
Yf = fft(yn.*hann(length(yn))');
figure(2)
subplot(211)
plot(fx, abs(Xf(1:length(fx))))
title('Xf');
xlabel('Frequency');
ylabel('Amplitude');
subplot(212)
plot(fy, abs(Yf(1:length(fy))))
title('Yf');
xlabel('Frequency');
ylabel('Amplitude');