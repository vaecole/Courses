% read dat.bin
fid=fopen('dat.bin');
raw_data = fread(fid, Inf, 'uint8');
fclose(fid);

% recover voltage signals
Vref=4.5;
x=zeros(1800,0);
for i = 0:1799
    bin_x = '';
    for j = 1:3
        bin_x = strcat(bin_x, dec2bin(raw_data(i*3+j), 8));
    end
    symbol = 1;
    if bin_x(1) == '1'
        symbol = -1;
    end
    % with symbol, bit invert, +1, get the original decimal value
    dec = symbol*((power(2, 23)-1-bin2dec(bin_x(2:24))+1));
    % calculate voltage by Vref, pear off gain, and rescale to microvolt(µV)
    x(i+1) = power(10,6)*dec*Vref/power(2, 23)/24;
end

% plot time-domain waveform
Fs = 250;
N = length(x);
t = (0:N-1)/Fs;
plot(t,x(1:N),'g');
axis([0 max(t) min(x)-0.1 max(x)+0.1]);
xlabel('Time/(s)');
ylabel('Voltage/(µV)');
title('Time-Domain Waveform of Voltage Signal');

