Xf = fft(xn);
Hf = fft(hn);
Yf = Xf.*Hf;
yn2 = ifft(Yf);
plot(yn2)
legend('yn2')
title('Using DFT and IDFT')
