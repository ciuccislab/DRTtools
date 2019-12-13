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

% default values
rbf_gaussian_4_FWHM = @(x) exp(-(x).^2)-1/2;
rbf_C2_matern_4_FWHM = @(x) exp(-abs(x)).*(1+abs(x))-1/2;
rbf_C4_matern_4_FWHM = @(x) 1/3*exp(-abs(x)).*(3+3*abs(x)+abs(x).^2)-1/2;
rbf_C6_matern_4_FWHM = @(x) 1/15*exp(-abs(x)).*(15+15*abs(x)+6*abs(x).^2+abs(x).^3)-1/2;
rbf_inverse_quadratic_4_FWHM = @(x)  1./(1+(x).^2)-1/2;
rbf_inverse_quadric_4_FWHM = @(x)  1./sqrt(1+(x).^2)-1/2;
rbf_cauchy_4_FWHM = @(x)  1./(1+abs(x))-1/2; 

handles.FWHM_gaussian = 2*fzero(@(x) rbf_gaussian_4_FWHM(x), 1);
handles.FWHM_C2_matern = 2*fzero(@(x) rbf_C2_matern_4_FWHM(x), 1);
handles.FWHM_C4_matern = 2*fzero(@(x) rbf_C4_matern_4_FWHM(x), 1);
handles.FWHM_C6_matern = 2*fzero(@(x) rbf_C6_matern_4_FWHM(x), 1);
handles.FWHM_inverse_quadratic = 2*fzero(@(x) rbf_inverse_quadratic_4_FWHM(x), 1);
handles.FWHM_inverse_quadric = 2*fzero(@(x) rbf_inverse_quadric_4_FWHM(x), 1);
handles.FWHM_cauchy = 2*fzero(@(x) rbf_cauchy_4_FWHM(x) ,1);

handles.FWHM_coeff = handles.FWHM_gaussian;
handles.rbf_type = 'gaussian';
handles.data_used = 'Combined Re-Im Data';
handles.integration_algorithm = 'integral';
handles.lambda = 1e-3;
handles.coeff = 0.5;
handles.file_type = 'Csv file';
handles.shape_control = 'Coefficient to FWHM';
handles.der_used = '1st-order';
handles.data_exist = false;
handles.use_induct = false;
% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = import1_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;

% --- Executes on button press in graph.
function graph_Callback(hObject, eventdata, handles)

h=figure;
saveas(handles.axes,h);


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

        handles.freq = A(:,1);
        handles.Z_prime_mat = A(:,2);
        handles.Z_double_prime_mat = A(:,3);
        handles.data_exist = true;

    otherwise
        warning('Invalid file type')
end

handles.freq_0 = handles.freq;
handles.Z_prime_mat_0 = handles.Z_prime_mat;
handles.Z_double_prime_mat_0 = handles.Z_double_prime_mat;

handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:); %this is the experimental impedance data

axes(handles.axes_panel_EIS)
plot(handles.Z_prime_mat,-handles.Z_double_prime_mat(:),'ob', 'MarkerSize', 3);
xlabel(handles.axes_panel_EIS,'$Z^{\prime}(f)$', 'Interpreter', 'Latex','Fontsize',24)
ylabel(handles.axes_panel_EIS,'$-Z^{\prime\prime}(f)$','Interpreter', 'Latex','Fontsize',24);
set(gca,'FontSize',20)
axis equal

set(handles.inductance,'Value',1)
set(handles.panel_EIS, 'Visible', 'on');
set(handles.panel_drt, 'Visible', 'off');

guidata(hObject,handles)

% --- Selecting the type of discretization
function dis_button_Callback(hObject, eventdata, handles)

switch get(handles.dis_button,'Value')
    case 1 % User selects Gaussian.
    handles.FWHM_coeff = handles.FWHM_gaussian;
    handles.rbf_type = 'gaussian';
    set(handles.RBF_option, 'Visible', 'on');
    % end select gaussian

    case 2 % User selects C2 Matern.
    handles.FWHM_coeff = handles.FWHM_C2_matern;
    handles.rbf_type = 'C2_matern';
    set(handles.RBF_option, 'Visible', 'on');
    % end select C2 Matern
    
    case 3 % User selects C4 Matern.
    handles.FWHM_coeff = handles.FWHM_C4_matern;
    handles.rbf_type = 'C4_matern';
    set(handles.RBF_option, 'Visible', 'on');
    % end select C4 Matern
    
    case 4 % User selects C6 Matern.
    handles.FWHM_coeff = handles.FWHM_C6_matern;
    handles.rbf_type = 'C6_matern';
    set(handles.RBF_option, 'Visible', 'on');
    % end select C6 Matern
    
    case 5 % User selects inverse_quadratic.
    handles.FWHM_coeff = handles.FWHM_inverse_quadratic;
    handles.rbf_type = 'inverse_quadratic';
    set(handles.RBF_option, 'Visible', 'on');
    % end select inverse_quadratic
    
    case 6 % User selects inverse_quadric.
    handles.FWHM_coeff = handles.FWHM_inverse_quadric;
    handles.rbf_type = 'inverse_quadric';
    set(handles.RBF_option, 'Visible', 'on');
    % end select inverse_quadric
    
    case 7 % User selects cauchy.
    handles.FWHM_coeff = handles.FWHM_cauchy;
    handles.rbf_type = 'cauchy';
    set(handles.RBF_option, 'Visible', 'on');
    % end select cauchy
    case 8
    handles.rbf_type = 'piecewise';
    set(handles.RBF_option, 'Visible', 'off');
    
    otherwise
end


guidata(hObject,handles) 


function dis_button_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function inductance_Callback(hObject, eventdata, handles)

if handles.data_exist
    switch get(handles.inductance,'Value')

        case 1 %keep data fitting without inductance
            handles.use_induct = false;

            handles.freq = handles.freq_0;
            handles.Z_prime_mat = handles.Z_prime_mat_0;
            handles.Z_double_prime_mat = handles.Z_double_prime_mat_0; 

            handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:);

        case 2 %keep data fitting with inductance
            handles.use_induct = true;

            handles.freq=handles.freq_0;
            handles.Z_prime_mat=handles.Z_prime_mat_0;
            handles.Z_double_prime_mat=handles.Z_double_prime_mat_0; 

            handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:);

        case 3 %discard data
            handles.use_induct = false;
            
            is_neg = -handles.Z_double_prime_mat(:)<0;
            index = find(is_neg==1);
            handles.Z_double_prime_mat(index) = [];
            handles.Z_prime_mat(index) = [];
            handles.freq(index)=[];

        otherwise
    end
    
   
    handles.Z_exp = handles.Z_prime_mat(:)+ i*handles.Z_double_prime_mat(:);

    axes(handles.axes_panel_EIS)
    plot(handles.Z_prime_mat,-handles.Z_double_prime_mat(:),'ob', 'MarkerSize', 3);
    xlabel(handles.axes_panel_EIS,'$Z^{\prime}(f)$', 'Interpreter', 'Latex','Fontsize',24);
    ylabel(handles.axes_panel_EIS,'$-Z^{\prime\prime}(f)$','Interpreter', 'Latex','Fontsize',24);
    set(gca,'FontSize',20)
    axis equal

end


guidata(hObject,handles) 


function inductance_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function derivative_Callback(hObject, eventdata, handles)

str = get(hObject, 'String');
val = get(hObject,'Value');

handles.der_used = str{val};

guidata(hObject,handles) 


function derivative_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function value_Callback(hObject, eventdata, handles)

 handles.lambda= str2num(get(handles.value,'String'));
 guidata(hObject,handles) 

function value_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
   
end
 

function plot_pop_Callback(hObject, eventdata, handles)

str = get(hObject, 'String');
val = get(hObject,'Value');

handles.data_used = str{val};

guidata(hObject,handles) 

function plot_pop_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function shape_Callback(hObject, eventdata, handles)

str = get(hObject, 'String');
val = get(hObject,'Value');

handles.shape_control = str{val};

guidata(hObject,handles) 

function shape_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    
end


function coef_Callback(hObject, eventdata, handles)

 handles.coeff= str2double(get(handles.coef,'String'));
 
 guidata(hObject,handles) 

 
function coef_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
     
end


function regularization_button_Callback(hObject, eventdata, handles)

% bounds ridge regression
    handles.lb_im = zeros(numel(handles.freq)+2,1);
    handles.ub_im = Inf*ones(numel(handles.freq)+2,1);
    handles.x_im_0 = ones(size(handles.lb_im));
    handles.lb_re = zeros(numel(handles.freq)+2,1);
    handles.ub_re = Inf*ones(numel(handles.freq)+2,1);
    handles.x_re_0 = ones(size(handles.lb_re));
    handles.taumax = ceil(max(log10(1./handles.freq)))+1;    
    handles.taumin = floor(min(log10(1./handles.freq)))-1;
    
    handles.options = optimset('algorithm','interior-point-convex','Display','off','TolFun',1e-15,'TolX',1e-10,'MaxFunEvals', 1E5);
    
    handles.b_re = real(handles.Z_exp);% experimental
    handles.b_im = -imag(handles.Z_exp);
    
if  strcmp(handles.rbf_type,'piecewise')

    handles.freq_out = handles.freq;
    
    handles.A_re = compute_A_re(handles.freq);
    handles.A_im = compute_A_im(handles.freq);
    handles.L_re = compute_L_re(handles.freq);
    handles.L_im = compute_L_im(handles.freq);
    handles.M_re = handles.L_re'*handles.L_re;
    handles.M_im = handles.L_im'*handles.L_im;
       
else

%   calculate the epsilon
    switch handles.shape_control
        case 'Coefficient to FWHM'
            handles.delta = mean(diff(log(1./handles.freq))); 
            handles.epsilon  = handles.coeff*handles.FWHM_coeff/handles.delta;
        case 'Shape Factor'
            handles.epsilon = handles.coeff;
    end
    
    handles.freq_out = logspace(-handles.taumin, -handles.taumax, 10*numel(handles.freq));
    handles.A_re = assemble_A_re(handles.freq, handles.epsilon, handles.rbf_type, handles.integration_algorithm);
    handles.A_im = assemble_A_im(handles.freq, handles.epsilon, handles.rbf_type, handles.integration_algorithm);
    handles.M_re = assemble_M_re(handles.freq, handles.epsilon, handles.rbf_type, handles.der_used);
    handles.M_im = assemble_M_im(handles.freq, handles.epsilon, handles.rbf_type, handles.der_used);
    
end

if  handles.use_induct
    handles.A_im(:,1) = -2*pi*(handles.freq(:));
end


[H_re,f_re] = quad_format(handles.A_re, handles.b_re, handles.M_re, handles.lambda);
[H_im,f_im] = quad_format(handles.A_im, handles.b_im, handles.M_im, handles.lambda);
[H_combined,f_combined] = quad_format_combined(handles.A_re, handles.A_im, handles.b_re, handles.b_im, handles.M_re, handles.M_im, handles.lambda);
warning('off')

axes(handles.axes_panel_drt)

switch handles.data_used
  
    case 'Combined Re-Im Data'

        handles.x_ridge_combined = quadprog(H_combined, f_combined, [], [], [], [], handles.lb_re, handles.ub_re, handles.x_re_0, handles.options);
        
            if strcmp(handles.rbf_type,'piecewise')
                
                    handles.gamma_ridge_combined = handles.x_ridge_combined(3:end);
                    
                    semilogx(1./handles.freq_out, handles.gamma_ridge_combined, '-k', 'LineWidth', 3);
                    hold on
                    semilogx(1./handles.freq_out, handles.gamma_ridge_combined, 'or', 'LineWidth', 3);
                    hold off
                    handles.rl=handles.x_ridge_combined(1:2);
                    handles.graph= handles.gamma_ridge_combined;
            else
                
                    handles.gamma_ridge_combined_fine = map_array_to_gamma(handles.freq_out, handles.freq, handles.x_ridge_combined(3:end), handles.epsilon, handles.rbf_type);
                    handles.gamma_ridge_combined_coarse = map_array_to_gamma(handles.freq, handles.freq, handles.x_ridge_combined(3:end), handles.epsilon, handles.rbf_type);

                    semilogx(1./handles.freq_out, handles.gamma_ridge_combined_fine, '-k', 'LineWidth', 3);
                    hold on
                    semilogx(1./handles.freq, handles.gamma_ridge_combined_coarse, 'or', 'LineWidth', 3);
                    hold off
                    handles.rl = handles.x_ridge_combined(1:2);
                    handles.graph = handles.gamma_ridge_combined_fine;
            end

    case 'Im Data'

        handles.x_ridge_im = quadprog(H_im, f_im, [], [], [], [], handles.lb_im, handles.ub_im, handles.x_im_0, handles.options);
        
            if strcmp(handles.rbf_type,'piecewise')
                
                    handles.gamma_ridge_im = handles.x_ridge_im(3:end);
                    
                    semilogx(1./handles.freq_out, handles.gamma_ridge_im, '-k', 'LineWidth', 3);
                    hold on
                    semilogx(1./handles.freq_out, handles.gamma_ridge_im, 'or', 'LineWidth', 3);
                    hold off
                    
                    handles.rl = handles.x_ridge_im(1:2);
                    handles.graph = handles.gamma_ridge_im;
            else
                
                    handles.gamma_ridge_im_fine = map_array_to_gamma(handles.freq_out, handles.freq, handles.x_ridge_im(3:end), handles.epsilon, handles.rbf_type);
                    handles.gamma_ridge_im_coarse = map_array_to_gamma(handles.freq, handles.freq, handles.x_ridge_im(3:end), handles.epsilon, handles.rbf_type);
                    
                    semilogx(1./handles.freq_out, handles.gamma_ridge_im_fine, '-k', 'LineWidth', 3);
                    hold on
                    semilogx(1./handles.freq, handles.gamma_ridge_im_coarse, 'or', 'LineWidth', 3);
                    hold off
                    
                    handles.rl = handles.x_ridge_im(1:2);
                    handles.graph = handles.gamma_ridge_im_fine;
            end

    case 'Re Data'

         handles.x_ridge_re = quadprog(H_re, f_re, [], [], [], [], handles.lb_re, handles.ub_re, handles.x_re_0, handles.options);

            if strcmp(handles.rbf_type,'piecewise')

                    handles.gamma_ridge_re = handles.x_ridge_re(3:end);
                    
                    semilogx(1./handles.freq_out, handles.gamma_ridge_re, '-k', 'LineWidth', 3);
                    hold on
                    semilogx(1./handles.freq_out, handles.gamma_ridge_re, 'or', 'LineWidth', 3);
                    hold off
                    
                    handles.rl = handles.x_ridge_re(1:2);
                    handles.graph = handles.gamma_ridge_re;
                    
            else
                
                    handles.gamma_ridge_re_fine = map_array_to_gamma(handles.freq_out, handles.freq, handles.x_ridge_re(3:end), handles.epsilon, handles.rbf_type);
                    handles.gamma_ridge_re_coarse = map_array_to_gamma(handles.freq, handles.freq, handles.x_ridge_re(3:end), handles.epsilon, handles.rbf_type);

                    semilogx(1./handles.freq_out, handles.gamma_ridge_re_fine, '-k', 'LineWidth', 3);
                    hold on
                    semilogx(1./handles.freq, handles.gamma_ridge_re_coarse, 'or', 'LineWidth', 3);
                    hold off
                    
                    handles.rl = handles.x_ridge_re(1:2);
                    handles.graph = handles.gamma_ridge_re_fine;
            end
end

xlabel(handles.axes_panel_drt,'$\tau/s$', 'Interpreter', 'Latex','Fontsize',24)
ylabel(handles.axes_panel_drt,'$\gamma(\tau)/\Omega$','Interpreter', 'Latex','Fontsize',24);
axis([10^(handles.taumin) 10^(handles.taumax) 0 (1.1*max(handles.graph))])
set(gca,'FontSize',20)
set(gca,'XTick',logspace(-10,10,11))

set(handles.panel_EIS, 'Visible', 'off');
set(handles.panel_drt, 'Visible', 'on');

guidata(hObject,handles)


function EIS_data_Callback(hObject, eventdata, handles)
%
if handles.data_exist

    axes(handles.axes_panel_EIS)
    plot(handles.Z_prime_mat,-handles.Z_double_prime_mat(:),'ob', 'MarkerSize', 3);
    xlabel(handles.axes_panel_EIS,'$Z^{\prime}(f)$', 'Interpreter', 'Latex','Fontsize',24)
    ylabel(handles.axes_panel_EIS,'$-Z^{\prime\prime}(f)$','Interpreter', 'Latex','Fontsize',24);
    set(gca,'FontSize',20)
    axis equal

end

set(handles.panel_EIS, 'Visible', 'on');
set(handles.panel_drt, 'Visible', 'off');


function deconvolved_DRT_Callback(hObject, eventdata, handles)
% 
set(handles.panel_EIS, 'Visible', 'off');
set(handles.panel_drt, 'Visible', 'on');


function Export_Callback(hObject, eventdata, handles)

startingFolder = 'C:\*';
if ~exist(startingFolder, 'dir')
	% If that folder doesn't exist, just start in the current folder.
	startingFolder = pwd;
end

[baseFileName, folder] = uiputfile({ '*.txt', 'txt files (*.txt)';'*.csv','csv files (*.csv)'}, 'Select a file');

fullFileName = fullfile(folder, baseFileName);

col_g = handles.graph(:);
col_t = 1./handles.freq_out(:);
col_g_1  = handles.rl(1);
col_t_1  = handles.rl(2);
fid  = fopen(fullFileName,'wt');
fprintf(fid,'%s, %e \n','L',col_g_1);
fprintf(fid,'%s, %e \n', 'R',col_t_1);
fprintf(fid,'%s, %s \n','gamma(tau)','tau');
fprintf(fid,'%e, %e \n', [col_g(:), col_t(:)].');
fclose(fid);

guidata(hObject,handles)


function Save_butt_Callback(hObject, eventdata, handles)

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

newfig_1=figure(); %new figure
copyobj(handles.axes_panel_drt, newfig_1);
set_size_fig
saveas(gcf,fullFileName)
close(gcf); 
    
 guidata(hObject,handles)
 
 
 function Save_butt_im_Callback(hObject, eventdata, handles)

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

newfig_1=figure(); %new figure
copyobj(handles.axes_panel_EIS, newfig_1);
set_size_fig
saveas(gcf,fullFileName)
close(gcf); 
    
 guidata(hObject,handles)

