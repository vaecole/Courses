hn = load('data/hn_q3.mat').hn;
Hf=fft(hn);
fs = 100;
figure(2)
subplot(211)
plot(20*log10(abs(Hf(1:fs/2))))
xlabel('Frequency')
ylabel('Amplitude(dB)')
subplot(212)
plot(unwrap(angle(Hf(1:fs/2))))
xlabel('Frequency')
ylabel('Phase')