hn = load('data/hn_q3.mat').hn;
xn = load('data/xn_q3.mat').xn;
yn = conv(xn, hn);
hold on;
plot(xn)
plot(yn)
legend('xn','yn');
hold off;