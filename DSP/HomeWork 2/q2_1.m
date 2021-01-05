xn = load('data/xn_q2.mat').x;
hn = load('data/hn_q2.mat').h;
yn1 = conv(xn, hn);
plot(yn1)
legend('yn1')
title('Convolution yn1 of xn and hn')
