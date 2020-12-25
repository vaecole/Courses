function spectrum_plot(f2, N, title_)
    f1=70;
    fs=2000;
    n=(0:N-1);
    x=sin(2*pi*f1*n/fs)+sin(2*pi*f2*n/fs);
    df = fs/N;
    f=(0:df:fs/2);
    X=fft(x);
    Amp = abs(X);
    plot(f, 20*log10(Amp(1:length(f))))
    tt =title(title_);
    tt.FontSize = 12;
end

