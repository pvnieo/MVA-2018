%% In this script, images and sigmas will be always of type 'double' between 0 and 1

% Location where MatConvNet is installed
MatConvNet='D:\PorgramFiles\matconvnet-1.0-beta25';
% MatConvNet='/tsi/zone/tp/TP/delires/matconvnet-1.0-beta25/';

% Location where PnP ADMM is installed
PnP=pwd;
% PnP = '/tsi/zone/tp/TP/delires/PnP_ADMM_Chan_v3'

%%%%%%%%%%%%%%%%%%%%
%% Initialization %%
%%%%%%%%%%%%%%%%%%%%

%addpath(fullfile(PnP,'minimizers/'));
addpath(fullfile(PnP,'denoisers/'));
addpath(fullfile(PnP,'utils/'));
rng(0)  % Random number generator seed

% Setup of matconvnet
run(fullfile(MatConvNet,'matlab/vl_setupnn.m'));
% If setup fails (errors) you may need to recompile it
% vl_compilenn;

% Images directory
image_dir = 'test_images/';
im_filenames = {
			 'barbara.tif'
			 %'boats.tif'
			 %'cameraman.tif'
			 %'house.tif'
			 %'lena256.tif'
			 %'peppers.tif'
			 };

% Load target image
filename = im_filenames{1};
target = double(imread(fullfile(image_dir, filename)));

% Normalize target image to [0,1]
m = min(target(:));
M = max(target(:));
target = (target-m)/(M-m);

% Parameters
lambda_list = [0.0001];
sigma = 5/255;
show_results = 1;
write_results = 1;  % Save results in subfolder 'results'
verbose = 1;


%%%%%%%%%%%%%%%%%%%%%
%% Select denoiser %%
%%%%%%%%%%%%%%%%%%%%%

denoiser_name = 'FFDNet';  % Options: BM3D, FFDNet

switch denoiser_name
	case 'BM3D'
		denoiser = @denoiser_BM3D;
    case 'FFDNet'
		denoiser = @denoiser_FFDNet;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select inverse problem %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

problem = 'deblurring';  % Options: deblurring, inpainting, superresolution

% Generate corrupted image and proximal operator of datafit term
switch problem

	case 'deblurring'
		% Blur kernel 
		h = fspecial('gaussian',[8 8],5);
		
		% Corrupted image
		%input = imfilter(target,h,'circular');
        input = imfilter(target,h);
        
        % Proximal operator
		prox_datafit = @(z, rho) prox_datafit_deblurring(z, input, rho, h);
        
        % Initialization
		init = imresize(input,1);

	case 'inpainting'
		% Mask
		p = 0.5;  % Proportion of missing pixels
		mask = rand(size(target))>=p;
		
		% Corrupted image
		input = mask.*target + sigma * randn(size(target));	 % Corrupted image
	
		% Proximal operator
		prox_datafit = @(z, rho) prox_datafit_inpainting(z, input, rho, mask);
		
		% Initialization (Shepard's 2D interpolation)
		init = shepard_initialize(input, mask, 10);
		
        % Alternative initializations
        % init = input;
        % init = zeros(size(input));
        % init = randn(size(input));
        
	case 'superresolution'
		% Blur kernel and downsampling factor
		h = fspecial('gaussian',[9 9],1);
		K = 2;  % Downsampling factor
		
		% Corrupted image
		input = imfilter(target,h,'circular');
		input = downsample2(input,K);
		input = input + sigma*randn(size(input));
		
		% Proximal operator
		prox_datafit = @(z, rho) prox_datafit_superresolution(z, input, rho, h, K);
		
		% Initialization
		init = imresize(input,K);
end


%%%%%%%%%%%%%%%
%% Main loop %%
%%%%%%%%%%%%%%%

% Verbose:
if verbose
	fprintf('\n-----------------------------------------------\n');
	fprintf('PROCESSING %s...\n', filename);
end
		
for lamb = lambda_list

	% Execute algorithm PnP ADMM
	if verbose
		fprintf('  -> PROBLEM: %s\n', problem);
		fprintf('  -> DENOISER: %s\n', denoiser_name);
		fprintf('  -> SIGMA = %.1f\n', 255*sigma);
		fprintf('  -> LAMBDA = %.4f\n', lamb);
		fprintf('-----------------------------------------------\n');
	end
	params = {};
	params.target = target;	 % To compute PSNR of each iteration
	params.verbose = verbose;
    params.rho0 = 0.5;

	% Main algorithm
	tic
	[output, info] = PnP_ADMM(init, prox_datafit, denoiser, lamb, params);
	total_time = toc;

	% Compute PSNR of result
	output_psnr = psnr(output, target, 1);

	% Show results
	if show_results
		figure, imshow(target), title('Ground truth')
		figure, imshow(input), title('Corrupted')
		figure, imshow(output), title(sprintf('Restored by PnP ADMM: PSNR = %.3f', output_psnr))
	end

	% Verbose
	if verbose
		fprintf('\n-----------------------------------------------\n');
		fprintf('RESULTS:\n');
		fprintf('Total time: %.2f\n', total_time);
		fprintf('Restored by PnP_ADMM: PSNR = %.3f\n', output_psnr);
		fprintf('-----------------------------------------------\n\n');
	end
	
	% Write results
	if write_results
		result_filename = sprintf('results/result_%s_%s_sigma%.2f_lambda%.4f_rho%.2f.png',...
	                                 problem, denoiser_name, 255*sigma, lamb, params.rho0);
		imwrite(output, result_filename);
	    fprintf('  --> Result saved in %s\n\n', result_filename);
	end
	
end
