function output = prox_datafit_deblurring(z, input, rho, h)

h = padarray(h,[124, 124],0,'both');
h = fftshift(h);
h_hat = fft2(h);

input_hat = fft2(input);
z_hat = fft2(z);

output = (conj(h_hat).*input_hat + rho*z_hat)./(h_hat.*conj(h_hat) + rho);
output = ifft2(output);


