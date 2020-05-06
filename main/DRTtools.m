%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIT License
%
% Copyright (c) 2020 ciuccislab
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function varargout = DRTtools(varargin)
% Begin initialization code - DO NOT EDIT

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @import1_OpeningFcn, ...
                   'gui_OutputFcn',  @import1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before import1 is made visible.
function import1_OpeningFcn(hObject, eventdata, handles, varargin)

if ~license('test', 'Optimization_Toolbox')

    error('***Optimization Toolbox licence is missing, DRTtools terminated***')
    close(DRTtools)
    
end
% Choose default command line output for import1
handles.output = hObject;

set(handles.dis_button,'Value',1)
set(handles.plot_pop,'Value',1)
set(handles.derivative,'Value',1)
set(handles.shape,'Value',1)
set(handles.value,'String','1E-3')
set(handles.coef,'String','0.5')
set(handles.inductance,'Value',1)
set(handles.panel_EIS, 'Visible', 'on');
set(handles.running_signal, 'Visible', 'off');

handles.rbf_type = 'Gaussian';
handles.data_used = 'Combined Re-Im Data';
handles.lambda = 1e-3;
handles.coeff = 0.5;
handles.file_type = 'Csv file';
handles.shape_control = 'FWHM Coefficient';
handles.der_used = '1st-order';
handles.data_exist = false;
handles.drt_computed = false;
handles.credibility = false; %%%%<---double check if we should delete this
handles.upper_bound_fine = 0;
% Update handles structure
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = import1_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;


% --- Executes on button press in graph.
function graph_Callback(hObject, eventdata, handles)

h=figure;
saveas(handles.axes,h);


% --- Import data button.
function import_button_Callback(hObject, eventdata, handles)

startingFolder = 'C:\*';
if ~exist(startingFolder, 'dir')
	% If that folder doesn't exist, just start in the current folder.
	startingFolder = pwd;
end

[baseFileName, folder] = uigetfile({ '*.mat; *.txt; *.csv*','Data files (*.mat, *.txt,*.csv)'}, 'Select a file');
    
fullFileName = fullfile(folder, baseFileName);

[folder,baseFileName,ext] = fileparts(fullFileName);

if ~baseFileName
            % User clicked the Cancel button.
    return;
end

switch ext

case '.mat' % User selects Mat files.

    storedStructure = load(fullFileName);

    handles.freq = storedStructure.freq;
    handles.Z_prime_mat = storedStructure.Z_prime;
    handles.Z_double_prime_mat = storedStructure.Z_double_prime;
    handles.data_exist = true;

case '.txt' % User selects Txt files.
%     change comma to dot if necessary
    fid  = fopen(fullFileName,'r');
    f1 = fread(fid,'*char')';
    fclose(fid);

    baseFileName = strrep(f1,',','.');
    fid  = fopen(fullFileName,'w');
    fprintf(fid,'%s',baseFileName);
    fclose(fid);

    A = dlmread(fullFileName);
    index = find(A(:,1)==0); % find incorrect columns
    A(index,:)=[];

    handles.freq = A(:,1);
    handles.Z_prime_mat = A(:,2);
    handles.Z_double_prime_mat = A(:,3);

%     change back dot to comma if necessary    
    fid  = fopen(fullFileName,'w');
    fprintf(fid,'%s',f1);
    fclose(fid);
    handles.data_exist = true;

case '.csv' % User selects csv.
    A = csvread(fullFileName);
    index = find(A(:,1)==0); % find incorrect columns
    A(index,:)=[];
    
    handles.freq = A(:,1);
    handles.Z_prime_mat = A(:,2);
    handles.Z_double_prime_mat = A(:,3);
    handles.data_exist = true;

otherwise
    warning('Invalid file type')
    handles.data_exist = false;
end

handles.freq_0 = handles.freq;
handles.Z_prime_mat_0 = handles.Z_prime_mat;
handles.Z_double_prime_mat_0 = handles.Z_double_prime_mat;

handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:); %this is the experimental impedance data

inductance_Callback(hObject, eventdata, handles)
EIS_data_Callback(hObject, eventdata, handles)

handles.drt_computed = false;

guidata(hObject,handles)


% --- Selecting the type of discretization
function dis_button_Callback(hObject, eventdata, handles)

str = get(handles.dis_button,'String');
val = get(handles.dis_button,'Value');

handles.rbf_type = str{val};

if strcmp(handles.rbf_type,'Piecewise linear')
    set(handles.RBF_option, 'Visible', 'off');
else
    set(handles.RBF_option, 'Visible', 'on');
end


guidata(hObject,handles) 


% --- Selecting treatment to the inductance data
function inductance_Callback(hObject, eventdata, handles)

if ~handles.data_exist
    return
end

switch get(handles.inductance,'Value')

case 1 %keep data fitting without inductance

    handles.freq = handles.freq_0;
    handles.Z_prime_mat = handles.Z_prime_mat_0;
    handles.Z_double_prime_mat = handles.Z_double_prime_mat_0; 

    handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:);

case 2 %keep data fitting with inductance

    handles.freq = handles.freq_0;
    handles.Z_prime_mat = handles.Z_prime_mat_0;
    handles.Z_double_prime_mat = handles.Z_double_prime_mat_0; 

    handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:);

case 3 %discard data

    is_neg = -handles.Z_double_prime_mat(:)<0;
    index = find(is_neg==1);
    handles.Z_double_prime_mat(index) = [];
    handles.Z_prime_mat(index) = [];
    handles.freq(index) = [];

end

handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:);

EIS_data_Callback(hObject, eventdata, handles)


guidata(hObject,handles) 


% --- Selecting the order of derivative for regularization
function derivative_Callback(hObject, eventdata, handles)

str = get(hObject,'String');
val = get(hObject,'Value');

handles.der_used = str{val};

guidata(hObject,handles) 


% --- Entering the regularization parameter
function value_Callback(hObject, eventdata, handles)

 handles.lambda = abs(str2num(get(handles.value,'String')));
 
 guidata(hObject,handles) 

 
% --- Selecting the kind of data used for fitting
function plot_pop_Callback(hObject, eventdata, handles)

str = get(hObject,'String');
val = get(hObject,'Value');

handles.data_used = str{val};

guidata(hObject,handles) 


% --- RBF shape control option
function shape_Callback(hObject, eventdata, handles)

str = get(hObject,'String');
val = get(hObject,'Value');

handles.shape_control = str{val};

guidata(hObject,handles) 


% --- Input to the RBF shape
function coef_Callback(hObject, eventdata, handles)

 handles.coeff = str2double(get(handles.coef,'String'));
 
 guidata(hObject,handles) 

 
% --- Simple Running regularization
function handles = regularization_button_Callback(hObject, eventdata, handles)

if ~handles.data_exist
    return
end
    
set(handles.running_signal, 'Visible', 'on');

% bounds ridge regression
handles.lb_im = zeros(numel(handles.freq)+2,1);
handles.ub_im = Inf*ones(numel(handles.freq)+2,1);
handles.x_im_0 = ones(size(handles.lb_im));
handles.lb_re = zeros(numel(handles.freq)+2,1);
handles.ub_re = Inf*ones(numel(handles.freq)+2,1);
handles.x_re_0 = ones(size(handles.lb_re));
handles.taumax = ceil(max(log10(1./handles.freq)))+0.5;    
handles.taumin = floor(min(log10(1./handles.freq)))-0.5;

handles.options = optimset('algorithm','interior-point-convex','Display','off','TolFun',1e-15,'TolX',1e-10,'MaxFunEvals', 1E5);

handles.b_re = real(handles.Z_exp);% experimental
handles.b_im = imag(handles.Z_exp);% negative sign deleted for the sake of consistency with python code

% compute epsilon
handles.epsilon = compute_epsilon(handles.freq, handles.coeff, handles.rbf_type, handles.shape_control);

% calculate the A_matrix and M_matrix
handles.freq_fine = logspace(-handles.taumin, -handles.taumax, 10*numel(handles.freq));
handles.A_re = assemble_A_re(handles.freq, handles.epsilon, handles.rbf_type);
handles.A_im = assemble_A_im(handles.freq, handles.epsilon, handles.rbf_type);
handles.M = assemble_M(handles.freq, handles.epsilon, handles.rbf_type, handles.der_used);
    
% adding the inductance column to the A_im_matrix if necessary
if  get(handles.inductance,'Value')==2
    
    handles.A_im(:,1) = 2*pi*(handles.freq(:));% negative sign deleted for the sake of consistency with python code
    
end

% preparing for quadratic programming
[H_re,f_re] = quad_format(handles.A_re, handles.b_re, handles.M, handles.lambda);
[H_im,f_im] = quad_format(handles.A_im, handles.b_im, handles.M, handles.lambda);
[H_combined,f_combined] = quad_format_combined(handles.A_re, handles.A_im, handles.b_re, handles.b_im, handles.M, handles.lambda);
warning('off')

axes(handles.axes_panel_drt)

% Running ridge regression
switch handles.data_used
  
case 'Combined Re-Im Data'

    handles.x_ridge = quadprog(H_combined, f_combined, [], [], [], [], handles.lb_re, handles.ub_re, handles.x_re_0, handles.options);

    %prepare for HMC sampler
    res_re = handles.A_re*handles.x_ridge-handles.b_re;
    res_im = handles.A_im*handles.x_ridge-handles.b_im;
    sigma_re_im = std([res_re;res_im]);

    inv_V = 1/sigma_re_im^2*eye(numel(handles.freq));
    
    Sigma_inv = (handles.A_re'*inv_V*handles.A_re) + (handles.A_im'*inv_V*handles.A_im) + (handles.lambda/sigma_re_im^2)*handles.M;%%<--look at the handles!
    handles.Sigma_inv = (Sigma_inv+Sigma_inv')/2;
    handles.mu = Sigma_inv\(handles.A_re'*inv_V*handles.b_re + handles.A_im'*inv_V*handles.b_im);

case 'Im Data'

    handles.x_ridge = quadprog(H_im, f_im, [], [], [], [], handles.lb_im, handles.ub_im, handles.x_im_0, handles.options);

    %prepare for HMC sampler
    res_im = handles.A_im*handles.x_ridge-handles.b_im;
    sigma_re_im = std(res_im);

    inv_V = 1/sigma_re_im^2*eye(numel(handles.freq));
    
    Sigma_inv = (handles.A_im'*inv_V*handles.A_im) + (handles.lambda/sigma_re_im^2)*handles.M;
    handles.Sigma_inv = (Sigma_inv+Sigma_inv')/2;
    handles.mu = Sigma_inv\(handles.A_im'*inv_V*handles.b_im);


case 'Re Data'

    handles.x_ridge = quadprog(H_re, f_re, [], [], [], [], handles.lb_re, handles.ub_re, handles.x_re_0, handles.options);

    %prepare for HMC sampler
    res_re = handles.A_re*handles.x_ridge-handles.b_re;
    sigma_re_im = std(res_re);
    
    inv_V = 1/sigma_re_im^2*eye(numel(handles.freq));
    
    Sigma_inv = (handles.A_re'*inv_V*handles.A_re) + (handles.lambda/sigma_re_im^2)*handles.M;
    handles.Sigma_inv = (Sigma_inv+Sigma_inv')/2;
    handles.mu = Sigma_inv\(handles.A_re'*inv_V*handles.b_re);
        
end
    
handles.upper_bound = 0;

handles.drt_computed = true;
handles.credibility = false;

handles = deconvolved_DRT_Callback(hObject, eventdata, handles);
set(handles.running_signal, 'Visible', 'off');

guidata(hObject,handles);


%%%--- Bayesian run
function bayesian_button_Callback(hObject, eventdata, handles)

if ~handles.data_exist
    return
end    

handles = regularization_button_Callback(hObject, eventdata, handles);

set(handles.running_signal, 'Visible', 'on');

%Running HMC sampler
handles.mu = handles.mu(3:end);
handles.Sigma_inv = handles.Sigma_inv(3:end,3:end);
handles.Sigma = inv(handles.Sigma_inv);

F = eye(numel(handles.x_ridge(3:end)));
g = eps*ones(size(handles.x_ridge(3:end)));
initial_X = handles.x_ridge(3:end)+100*eps;
L = str2num(get(handles.sample_number,'String'));

if L>=1000

    handles.Xs = HMC_exact(F, g, handles.Sigma, handles.mu, true, L, initial_X);

%     handles.lower_bound = quantile(handles.Xs(:,500:end),.005,2);
%     handles.upper_bound = quantile(handles.Xs(:,500:end),.995,2);
    handles.lower_bound = quantile_alter(handles.Xs(:,500:end),.005,2,'R-5');
    handles.upper_bound = quantile_alter(handles.Xs(:,500:end),.995,2,'R-5');
    handles.mean = mean(handles.Xs(:,500:end),2);
    
    set(handles.running_signal, 'Visible', 'off');
    handles.credibility = true;
    
else
    
    set(handles.running_signal, 'Visible', 'off');
    handles.credibility = false;    
    error('***Sample number less than 1000, the HMC sampler would not start***')
    
end

handles = deconvolved_DRT_Callback(hObject, eventdata, handles);

guidata(hObject,handles)


% --- Plotting EIS curve and switching to the plot
function EIS_data_Callback(hObject, eventdata, handles)

if handles.data_exist
    
    axes(handles.axes_panel_EIS)
    plot(handles.Z_prime_mat,-handles.Z_double_prime_mat(:),'ob', 'MarkerSize', 3);
    xlabel(handles.axes_panel_EIS,'$Z^{\prime}(f)$', 'Interpreter', 'Latex','Fontsize',24);
    ylabel(handles.axes_panel_EIS,'$-Z^{\prime\prime}(f)$','Interpreter', 'Latex','Fontsize',24);
    set(gca,'FontSize',20)
    axis equal

end

set(handles.panel_EIS, 'Visible', 'on');
set(handles.panel_drt, 'Visible', 'off');


% --- plotting the DRT
function handles = deconvolved_DRT_Callback(hObject, eventdata, handles)
% Running ridge regression

if handles.drt_computed
       
    if strcmp(handles.rbf_type,'Piecewise linear')
        
        handles.freq_fine = handles.freq;
        handles.gamma_ridge_fine = handles.x_ridge(3:end);
        
        if handles.credibility
            
            handles.gamma_mean_fine = handles.mean;
            handles.lower_bound_fine = handles.lower_bound;
            handles.upper_bound_fine = handles.upper_bound;
            
            ciplot(handles.lower_bound, handles.upper_bound, 1./handles.freq_fine, 0.7*[1 1 1]);%plot CI
            hold on
            plot(1./handles.freq_fine, handles.gamma_mean_fine, '-b', 'LineWidth', 3);%plot mean
            
        end
        
        plot(1./handles.freq_fine, handles.gamma_ridge_fine, '-k', 'LineWidth', 3);%plot MAP 
        hold off

    else %%% for RBF case

        if handles.credibility
            
            handles.gamma_mean_fine = map_array_to_gamma(handles.freq_fine, handles.freq, handles.mean, handles.epsilon, handles.rbf_type);
            handles.lower_bound_fine = map_array_to_gamma(handles.freq_fine, handles.freq, handles.lower_bound, handles.epsilon, handles.rbf_type);
            handles.upper_bound_fine = map_array_to_gamma(handles.freq_fine, handles.freq, handles.upper_bound, handles.epsilon, handles.rbf_type);
            
            ciplot(handles.lower_bound_fine, handles.upper_bound_fine, 1./handles.freq_fine, 0.7*[1 1 1]);%plot CI
            hold on
            plot(1./handles.freq_fine, handles.gamma_mean_fine, '-b', 'LineWidth', 3);%plot mean
                        
        end

        handles.gamma_ridge_fine = map_array_to_gamma(handles.freq_fine, handles.freq, handles.x_ridge(3:end), handles.epsilon, handles.rbf_type);
            
        plot(1./handles.freq_fine, handles.gamma_ridge_fine, '-k', 'LineWidth', 3);
        hold off

    end

    if handles.credibility
        %add legend
        h = legend('CI', 'Mean', 'MAP', 'Location','NorthWest');
        set(h,'Interpreter', 'LaTex','Fontsize', 24)
        legend boxoff
        
    end
    
%adding labels
xlabel(handles.axes_panel_drt,'$\tau/s$', 'Interpreter', 'Latex','Fontsize',24)
ylabel(handles.axes_panel_drt,'$\gamma(\tau)/\Omega$','Interpreter', 'Latex','Fontsize',24);

set(gca,'xscale','log','xlim',[10^(handles.taumin), 10^(handles.taumax)],'ylim',[0, 1.1*max([handles.gamma_ridge_fine;handles.upper_bound_fine])],'Fontsize',20,'xtick',10.^[-10:2:10])

end

set(handles.panel_EIS, 'Visible', 'off');
set(handles.panel_drt, 'Visible', 'on');

guidata(hObject,handles)


% --- Exporting the DRT data
function Export_Callback(hObject, eventdata, handles)

startingFolder = 'C:\*';
if ~exist(startingFolder, 'dir')
	% If that folder doesn't exist, just start in the current folder.
	startingFolder = pwd;

end

[baseFileName, folder] = uiputfile({ '*.txt', 'txt files (*.txt)';'*.csv','csv files (*.csv)'}, 'Select a file');

fullFileName = fullfile(folder, baseFileName);

fid  = fopen(fullFileName,'wt');

fprintf(fid,'%s, %e \n','L',handles.x_ridge(1));
fprintf(fid,'%s, %e \n','R',handles.x_ridge(2));

col_tau = 1./handles.freq_fine(:);

if ~handles.credibility % not bayesian
    
    col_gamma = handles.gamma_ridge_fine(:);
    fprintf(fid,'%s, %s \n','tau','gamma(tau)');
    fprintf(fid,'%e, %e \n', [col_tau(:), col_gamma(:)]');

else
    
    col_gamma = handles.gamma_ridge_fine(:);
    col_mean = handles.gamma_mean_fine(:);
    col_upper = handles.upper_bound_fine(:);
    col_lower = handles.lower_bound_fine(:);
    fprintf(fid,'%s, %s, %s, %s, %s\n', 'tau', 'MAP', 'Mean', 'Upperbound', 'Lowerbound');
    fprintf(fid,'%e, %e, %e, %e, %e\n', [col_tau(:), col_gamma(:), col_mean(:), col_upper(:), col_lower(:)]');

end

fclose(fid);

guidata(hObject,handles)


% --- Exporting the DRT figure
function Save_DRT_figure_Callback(hObject, eventdata, handles)

startingFolder = 'C:\*';
if ~exist(startingFolder, 'dir')
	% If that folder doesn't exist, just start in the current folder.
	startingFolder = pwd;
end

defaultFileName = fullfile(startingFolder, '*.fig');
[baseFileName, folder] = uiputfile(defaultFileName, 'Name a file');


if baseFileName == 0
    % User clicked the Cancel button.
    return;
end
            
fullFileName = fullfile(folder, baseFileName);

newfig_1 = figure(); %new figure
copyobj(handles.axes_panel_drt, newfig_1);
set_size_fig
saveas(gcf,fullFileName)
close(gcf); 
    
 guidata(hObject,handles)
 
 % --- Exporting the EIS figure
 function Save_EIS_figure_Callback(hObject, eventdata, handles)

startingFolder = 'C:\*';
if ~exist(startingFolder, 'dir')
	% If that folder doesn't exist, just start in the current folder.
	startingFolder = pwd;
end

defaultFileName = fullfile(startingFolder, '*.fig');
[baseFileName, folder] = uiputfile(defaultFileName, 'Name a file');


if baseFileName == 0
    % User clicked the Cancel button.
    return;
end
            
fullFileName = fullfile(folder, baseFileName);

newfig_1 = figure(); %new figure
copyobj(handles.axes_panel_EIS, newfig_1);
set_size_fig
saveas(gcf,fullFileName)
close(gcf); 
    
 guidata(hObject,handles)

