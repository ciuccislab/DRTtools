%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2021, Bruno Melo (B.M.G. Melo)
% All rights reserved.
% 
%
% Redistribution and use in source and binary forms, with or without modification, are permitted provided 
% that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
%    and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse 
%    or promote products derived from this software without specific prior written permission.
% 
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function varargout = DFRTtoEIS_nw(varargin)
% DFRT_MINE MATLAB code for DFRTtoEIS_nw.fig
%      DFRT_MINE, by itself, creates a new DFRT_MINE or raises the existing
%      singleton*.
%
%      H = DFRT_MINE returns the handle to a new DFRT_MINE or the handle to
%      the existing singleton*.
%
%      DFRT_MINE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DFRT_MINE.M with the given input arguments.
%
%      DFRT_MINE('Property','Value',...) creates a new DFRT_MINE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DFRTtoEIS_nw_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DFRTtoEIS_nw_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DFRTtoEIS_nw

% Last Modified by GUIDE v2.5 27-Apr-2020 17:07:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @DFRTtoEIS_nw_OpeningFcn, ...
    'gui_OutputFcn',  @DFRTtoEIS_nw_OutputFcn, ...
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


% --- Executes just before DFRTtoEIS_nw is made visible.
function DFRTtoEIS_nw_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DFRTtoEIS_nw (see VARARGIN)

% Choose default command line output for DFRTtoEIS_nw
handles.output = hObject;

% Update handles structure

matrix_gaussian_fit=0;
handles.residuals=[];
handles.matrix_gaussian_fit=matrix_gaussian_fit;
guidata(hObject, handles);


% UIWAIT makes DFRTtoEIS_nw wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DFRTtoEIS_nw_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in savedata.
function savedata_Callback(hObject, eventdata, handles)
% hObject    handle to savedata (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


f_dfrt=evalin('base','f_dfrt');
f_exp=evalin('base','f_exp');
X_DFRT_long = evalin('base', 'X_DFRT_long');
Y_DFRT_long = evalin('base', 'Y_DFRT_long');
Z_re_exp = evalin('base', 'Z_re_exp');
Z_imag_exp = evalin('base', 'Z_imag_exp');
Z_re_DFRT = evalin('base', 'Z_re_DFRT');
Z_imag_DFRT = evalin('base', 'Z_imag_DFRT');


N=length(X_DFRT_long)-length(Z_re_exp);
t=zeros(N,1);
Matrix1=[Z_re_exp; t];
Matrix2=[Z_imag_exp; t];
Matrixf_exp=[f_exp; t];



if length(f_dfrt)~=length(X_DFRT_long)
    
    NNN=length(X_DFRT_long)-length(f_dfrt);
    ttt=zeros(NNN,1);
    f_dfrt=[f_dfrt; ttt];
    
    Z_re_DFRT=[Z_re_DFRT; ttt];
    Z_imag_DFRT=[Z_imag_DFRT; ttt];
end


if evalin( 'base', 'exist(''GaussResults'',''var'') == 0' )         %without Gaussian Fits
    
    if (isempty(handles.residuals)==1)                              %without residuals
        
        Matrix_f = [Matrixf_exp Matrix1 Matrix2 f_dfrt Z_re_DFRT Z_imag_DFRT X_DFRT_long Y_DFRT_long];
        message = sprintf('Data will be saved in the following order:\nColumn 1: F_Exp\nColumn 2: Z_real_exp\nColumn 3: Z_imag_exp\nColumn 4: F_dfrt\nColumn 5: Z_real_dfrt\nColumn 6: Z_imag_dfrt\nColumn 7: tau\nColumn 8: DFRT');
        uiwait(helpdlg(message,'Saved data'));
    else                                                            %with residuals
        MatrixRfreq=[handles.residuals(:,1); t];
        MatrixRreal=[handles.residuals(:,2); t];
        MatrixRimag=[handles.residuals(:,3); t];
        MatrixR_f = [MatrixRfreq MatrixRreal MatrixRimag];
        Matrix_f = [Matrixf_exp Matrix1 Matrix2 f_dfrt Z_re_DFRT Z_imag_DFRT X_DFRT_long Y_DFRT_long MatrixR_f];
        message = sprintf('Data will be saved in the following order:\nColumn 1: F_Exp\nColumn 2: Z_real_exp\nColumn 3: Z_imag_exp\nColumn 4: F_dfrt\nColumn 5: Z_real_dfrt\nColumn 6: Z_imag_dfrt\nColumn 7: tau\nColumn 8: DFRT\nLast 3 columns : Residuals');
        uiwait(helpdlg(message,'Saved data'));
    end
elseif evalin( 'base', 'exist(''GaussResults'',''var'') == 1' )            %with Gaussian Fits
    
    Results = evalin('base', 'GaussResults');
    
    NNN=length(X_DFRT_long)-length(Results);
    ttt=zeros(NNN,1);
    Results=[Results; ttt];

    
    if (isempty(handles.residuals)==1)                                      %with Gaussian Fits and without residuals
        
        Matrix_f = [Matrixf_exp Matrix1 Matrix2 f_dfrt Z_re_DFRT Z_imag_DFRT X_DFRT_long Y_DFRT_long handles.MatrixGaussXY Results];
        message = sprintf('Data will be saved in the following order:\nColumn 1: F_Exp\nColumn 2: Z_real_exp\nColumn 3: Z_imag_exp\nColumn 4: F_dfrt\nColumn 5: Z_real_dfrt\nColumn 6: Z_imag_dfrt\nColumn 7: tau\nColumn 8: DFRT\nColumn 9-end: Gauss Fits');
        uiwait(helpdlg(message,'Saved data'));
    else                                                                    %with Gaussian Fits and with residuals
        MatrixRfreq=[handles.residuals(:,1); t];
        MatrixRreal=[handles.residuals(:,2); t];
        MatrixRimag=[handles.residuals(:,3); t];
        MatrixR_f = [MatrixRfreq MatrixRreal MatrixRimag];
        Matrix_f = [Matrixf_exp Matrix1 Matrix2 f_dfrt Z_re_DFRT Z_imag_DFRT X_DFRT_long Y_DFRT_long handles.MatrixGaussXY Results MatrixR_f];
        message = sprintf('Data will be saved in the following order:\nColumn 1: F_Exp\nColumn 2: Z_real_exp\nColumn 3: Z_imag_exp\nColumn 4: F_dfrt\nColumn 5: Z_real_dfrt\nColumn 6: Z_imag_dfrt\nColumn 7: tau\nColumn 8: DFRT\nColumn 9-: Gauss Fits\nLast 3 columns : Residuals');
        uiwait(helpdlg(message,'Saved data'));
    end
    
end

[filename, pathname] = uiputfile('.txt');
fullname = fullfile(pathname,filename);
dlmwrite(fullname,Matrix_f,'delimiter','\t');

matrix_gaussian_fit=0;
handles.matrix_gaussian_fit=matrix_gaussian_fit;
guidata(hObject, handles);



% --- Executes on button press in calc.
function calc_Callback(hObject, eventdata, handles)
% hObject    handle to calc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


gaussian_terms = str2num(get(handles.gaussian_terms,'String'));
Rinf=evalin('base','Rinf');
X_DFRT_long = evalin('base', 'X_DFRT_long');
Y_DFRT_long = evalin('base', 'Y_DFRT_long');
Z_re_exp = evalin('base', 'Z_re_exp');
Z_imag_exp = evalin('base', 'Z_imag_exp');
Z_re_DFRT = evalin('base', 'Z_re_DFRT');
Z_imag_DFRT = evalin('base', 'Z_imag_DFRT');


X=X_DFRT_long;
Y=Y_DFRT_long;
x_gaussian=log10(X);
y_gaussian=Y;



if (get(handles.edited_range,'value')==1)       %restricted range=on
    
    if gaussian_terms==1
        
        
        
        lb=ginput(3);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        XX=[log10(X(indexAtMin1))];
        YY=[Y(indexAtMin1)];
        
        diffsx1 = abs(X - lb(2,1));
        [minDiffx1, indexAtMinx1] = min(diffsx1);
        diffsx2 = abs(X - lb(3,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        
        if indexAtMinx1<indexAtMinx2
            x_gaussian=log10(X(indexAtMinx1:indexAtMinx2));
            y_gaussian=Y(indexAtMinx1:indexAtMinx2);
        else
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx1));
            y_gaussian=Y(indexAtMinx2:indexAtMinx1);
        end
        
        
        [fitresult, gof] = GaussianFit1(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit_x=X;
        [maxYValue, indexAtMaxY] = max(Gaussian_fit);
        xValueAtMaxYValue = Gaussian_fit_x(indexAtMaxY(1));
        Resistance = trapz(log(X), Gaussian_fit);
        
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        area(X, Gaussian_fit,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance; xValueAtMaxYValue; xValueAtMaxYValue/Resistance];
        
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
    elseif gaussian_terms==2
        
        lb=ginput(4);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2))];
        YY=[Y(indexAtMin1),Y(indexAtMin2)];
        diffsx1 = abs(X - lb(3,1));
        [minDiffx1, indexAtMinx1] = min(diffsx1);
        diffsx2 = abs(X - lb(4,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        
        if indexAtMinx1<indexAtMinx2
            x_gaussian=log10(X(indexAtMinx1:indexAtMinx2));
            y_gaussian=Y(indexAtMinx1:indexAtMinx2);
        else
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx1));
            y_gaussian=Y(indexAtMinx2:indexAtMinx1);
        end
        
        
        [fitresult, gof] = GaussianFit2(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        
        [maxYValue1, indexAtMaxY1] = max(Gaussian_fit1);
        xValueAtMaxYValue1 = X(indexAtMaxY1(1));
        Resistance1 = trapz(log(X), Gaussian_fit1);
        
        
        [maxYValue2, indexAtMaxY2] = max(Gaussian_fit2);
        xValueAtMaxYValue2 = X(indexAtMaxY2(1));
        Resistance2 = trapz(log(X), Gaussian_fit2);
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance1; xValueAtMaxYValue1; xValueAtMaxYValue1/Resistance1; Resistance2; xValueAtMaxYValue2; xValueAtMaxYValue2/Resistance2];
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1','R Gauss2','Tau2','Capacitance2'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        
        Gaussian_fit=[Gaussian_fit1 Gaussian_fit2];
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
    elseif gaussian_terms==3
        
        lb=ginput(5);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        diffs3 = abs(X - lb(3,1));
        [minDiff3, indexAtMin3] = min(diffs3);
        diffsx1 = abs(X - lb(4,1));
        [minDiffx1, indexAtMinx1] = min(diffsx1);
        diffsx2 = abs(X - lb(5,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        
        if indexAtMinx1<indexAtMinx2
            x_gaussian=log10(X(indexAtMinx1:indexAtMinx2));
            y_gaussian=Y(indexAtMinx1:indexAtMinx2);
        else
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx1));
            y_gaussian=Y(indexAtMinx2:indexAtMinx1);
        end
        
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2)),log10(X(indexAtMin3))];
        YY=[Y(indexAtMin1),Y(indexAtMin2),Y(indexAtMin3)];
        
        [fitresult, gof] = GaussianFit3(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        Gaussian_fit3=coef(7).*exp(-((log10(X)-coef(8))/coef(9)).^2);
        Gaussian_fit_x=X;
        [maxYValue, indexAtMaxY] = max(Gaussian_fit1);
        xValueAtMaxYValue = Gaussian_fit_x(indexAtMaxY(1));
        
        Resistance1 = trapz(log(X), Gaussian_fit1);
        Resistance2 = trapz(log(X), Gaussian_fit2);
        Resistance3 = trapz(log(X), Gaussian_fit3);
        [maxYValue1, indexAtMaxY1] = max(Gaussian_fit1);
        [maxYValue2, indexAtMaxY2] = max(Gaussian_fit2);
        [maxYValue3, indexAtMaxY3] = max(Gaussian_fit3);
        xValueAtMaxYValue1 = X(indexAtMaxY1(1));
        xValueAtMaxYValue2 = X(indexAtMaxY2(1));
        xValueAtMaxYValue3 = X(indexAtMaxY3(1));
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        area(X, Gaussian_fit3,'EdgeAlpha',0,'FaceColor',[255/255 180/255 143/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        
        Gaussian_fit=[Gaussian_fit1 Gaussian_fit2 Gaussian_fit3];
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance1; xValueAtMaxYValue1; xValueAtMaxYValue1/Resistance1; Resistance2; xValueAtMaxYValue2; xValueAtMaxYValue2/Resistance2; Resistance3; xValueAtMaxYValue3; xValueAtMaxYValue3/Resistance3];
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1','R Gauss2','Tau2','Capacitance2','R Gauss3','Tau3','Capacitance3'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
    elseif gaussian_terms==4
        
        
        
        lb=ginput(6);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        
        diffs3 = abs(X - lb(3,1));
        [minDiff3, indexAtMin3] = min(diffs3);
        
        diffs4 = abs(X - lb(4,1));
        [minDiff4, indexAtMin4] = min(diffs4);
        diffsx1 = abs(X - lb(5,1));
        [minDiffx1, indexAtMinx1] = min(diffsx1);
        diffsx2 = abs(X - lb(6,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        
        if indexAtMinx1<indexAtMinx2
            x_gaussian=log10(X(indexAtMinx1:indexAtMinx2));
            y_gaussian=Y(indexAtMinx1:indexAtMinx2);
        else
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx1));
            y_gaussian=Y(indexAtMinx2:indexAtMinx1);
        end
        
        
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2)),log10(X(indexAtMin3)),log10(X(indexAtMin4))];
        YY=[Y(indexAtMin1),Y(indexAtMin2),Y(indexAtMin3),Y(indexAtMin3)];
        
        [fitresult, gof] = GaussianFit4(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        Gaussian_fit3=coef(7).*exp(-((log10(X)-coef(8))/coef(9)).^2);
        Gaussian_fit4=coef(10).*exp(-((log10(X)-coef(11))/coef(12)).^2);
        Gaussian_fit_x=X;
        [maxYValue, indexAtMaxY] = max(Gaussian_fit1);
        xValueAtMaxYValue = Gaussian_fit_x(indexAtMaxY(1));
        
        Resistance1 = trapz(log(X), Gaussian_fit1);
        Resistance2 = trapz(log(X), Gaussian_fit2);
        Resistance3 = trapz(log(X), Gaussian_fit3);
        Resistance4 = trapz(log(X), Gaussian_fit4);
        [maxYValue1, indexAtMaxY1] = max(Gaussian_fit1);
        [maxYValue2, indexAtMaxY2] = max(Gaussian_fit2);
        [maxYValue3, indexAtMaxY3] = max(Gaussian_fit3);
        [maxYValue4, indexAtMaxY4] = max(Gaussian_fit4);
        xValueAtMaxYValue1 = X(indexAtMaxY1(1));
        xValueAtMaxYValue2 = X(indexAtMaxY2(1));
        xValueAtMaxYValue3 = X(indexAtMaxY3(1));
        xValueAtMaxYValue4 = X(indexAtMaxY4(1));
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        area(X, Gaussian_fit3,'EdgeAlpha',0,'FaceColor',[255/255 180/255 143/255])
        area(X, Gaussian_fit4,'EdgeAlpha',0,'FaceColor',[221/255 221/255 221/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        Gaussian_fit=[Gaussian_fit1 Gaussian_fit2 Gaussian_fit3 Gaussian_fit4];
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance1; xValueAtMaxYValue1; xValueAtMaxYValue1/Resistance1; Resistance2; xValueAtMaxYValue2; xValueAtMaxYValue2/Resistance2; Resistance3; xValueAtMaxYValue3; xValueAtMaxYValue3/Resistance3; Resistance4; xValueAtMaxYValue4; xValueAtMaxYValue4/Resistance4];  
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1','R Gauss2','Tau2','Capacitance2','R Gauss3','Tau3','Capacitance3','R Gauss4','Tau4','Capacitance4'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
    end
    
else                    %restricted range off
    
    if gaussian_terms==1
        
        
        
        lb=ginput(1);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        XX=[log10(X(indexAtMin1))];
        YY=[Y(indexAtMin1)];
        
        [fitresult, gof] = GaussianFit1(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit_x=X;
        [maxYValue, indexAtMaxY] = max(Gaussian_fit);
        xValueAtMaxYValue = Gaussian_fit_x(indexAtMaxY(1));
        Resistance = trapz(log(X), Gaussian_fit);
        
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance; xValueAtMaxYValue; xValueAtMaxYValue/Resistance];
       
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
    elseif gaussian_terms==2
        
        lb=ginput(2);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        
        
        
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2))];
        YY=[Y(indexAtMin1),Y(indexAtMin2)];
        
        [fitresult, gof] = GaussianFit2(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        
        [maxYValue1, indexAtMaxY1] = max(Gaussian_fit1);
        xValueAtMaxYValue1 = X(indexAtMaxY1(1));
        Resistance1 = trapz(log(X), Gaussian_fit1);
        
        
        [maxYValue2, indexAtMaxY2] = max(Gaussian_fit2);
        xValueAtMaxYValue2 = X(indexAtMaxY2(1));
        Resistance2 = trapz(log(X), Gaussian_fit2);
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance1; xValueAtMaxYValue1; xValueAtMaxYValue1/Resistance1; Resistance2; xValueAtMaxYValue2; xValueAtMaxYValue2/Resistance2];
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1','R Gauss2','Tau2','Capacitance2'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        Gaussian_fit=[Gaussian_fit1 Gaussian_fit2];
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
    elseif gaussian_terms==3
        
        
        
        lb=ginput(3);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        
        diffs3 = abs(X - lb(3,1));
        [minDiff3, indexAtMin3] = min(diffs3);
        
        
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2)),log10(X(indexAtMin3))];
        YY=[Y(indexAtMin1),Y(indexAtMin2),Y(indexAtMin3)];
        
        [fitresult, gof] = GaussianFit3(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        Gaussian_fit3=coef(7).*exp(-((log10(X)-coef(8))/coef(9)).^2);
        Gaussian_fit_x=X;
        [maxYValue, indexAtMaxY] = max(Gaussian_fit1);
        xValueAtMaxYValue = Gaussian_fit_x(indexAtMaxY(1));
        
        Resistance1 = trapz(log(X), Gaussian_fit1);
        Resistance2 = trapz(log(X), Gaussian_fit2);
        Resistance3 = trapz(log(X), Gaussian_fit3);
        [maxYValue1, indexAtMaxY1] = max(Gaussian_fit1);
        [maxYValue2, indexAtMaxY2] = max(Gaussian_fit2);
        [maxYValue3, indexAtMaxY3] = max(Gaussian_fit3);
        xValueAtMaxYValue1 = X(indexAtMaxY1(1));
        xValueAtMaxYValue2 = X(indexAtMaxY2(1));
        xValueAtMaxYValue3 = X(indexAtMaxY3(1));
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        area(X, Gaussian_fit3,'EdgeAlpha',0,'FaceColor',[255/255 180/255 143/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        Gaussian_fit=[Gaussian_fit1 Gaussian_fit2 Gaussian_fit3];
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance1; xValueAtMaxYValue1; xValueAtMaxYValue1/Resistance1; Resistance2; xValueAtMaxYValue2; xValueAtMaxYValue2/Resistance2; Resistance3; xValueAtMaxYValue3; xValueAtMaxYValue3/Resistance3];     
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1','R Gauss2','Tau2','Capacitance2','R Gauss3','Tau3','Capacitance3'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
    elseif gaussian_terms==4
        
        
        
        lb=ginput(4);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        
        diffs3 = abs(X - lb(3,1));
        [minDiff3, indexAtMin3] = min(diffs3);
        
        diffs4 = abs(X - lb(4,1));
        [minDiff4, indexAtMin4] = min(diffs4);
        
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2)),log10(X(indexAtMin3)),log10(X(indexAtMin4))];
        YY=[Y(indexAtMin1),Y(indexAtMin2),Y(indexAtMin3),Y(indexAtMin3)];
        
        [fitresult, gof] = GaussianFit4(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        Gaussian_fit3=coef(7).*exp(-((log10(X)-coef(8))/coef(9)).^2);
        Gaussian_fit4=coef(10).*exp(-((log10(X)-coef(11))/coef(12)).^2);
        Gaussian_fit_x=X;
        [maxYValue, indexAtMaxY] = max(Gaussian_fit1);
        xValueAtMaxYValue = Gaussian_fit_x(indexAtMaxY(1));
        
        Resistance1 = trapz(log(X), Gaussian_fit1);
        Resistance2 = trapz(log(X), Gaussian_fit2);
        Resistance3 = trapz(log(X), Gaussian_fit3);
        Resistance4 = trapz(log(X), Gaussian_fit4);
        [maxYValue1, indexAtMaxY1] = max(Gaussian_fit1);
        [maxYValue2, indexAtMaxY2] = max(Gaussian_fit2);
        [maxYValue3, indexAtMaxY3] = max(Gaussian_fit3);
        [maxYValue4, indexAtMaxY4] = max(Gaussian_fit4);
        xValueAtMaxYValue1 = X(indexAtMaxY1(1));
        xValueAtMaxYValue2 = X(indexAtMaxY2(1));
        xValueAtMaxYValue3 = X(indexAtMaxY3(1));
        xValueAtMaxYValue4 = X(indexAtMaxY4(1));
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        area(X, Gaussian_fit3,'EdgeAlpha',0,'FaceColor',[255/255 180/255 143/255])
        area(X, Gaussian_fit4,'EdgeAlpha',0,'FaceColor',[221/255 221/255 221/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);

        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        Gaussian_fit=[Gaussian_fit1 Gaussian_fit2 Gaussian_fit3 Gaussian_fit4];
        Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance1; xValueAtMaxYValue1; xValueAtMaxYValue1/Resistance1; Resistance2; xValueAtMaxYValue2; xValueAtMaxYValue2/Resistance2; Resistance3; xValueAtMaxYValue3; xValueAtMaxYValue3/Resistance3; Resistance4; xValueAtMaxYValue4; xValueAtMaxYValue4/Resistance4];       
        CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1','Tau1','Capacitance1','R Gauss2','Tau2','Capacitance2','R Gauss3','Tau3','Capacitance3','R Gauss4','Tau4','Capacitance4'}';
        CC2=num2cell(Plot_Results);
        CCC=[CC1 CC2];
        set(handles.table_results, 'data', CCC);           %Fill table
        handles.Gaussian_fit=Gaussian_fit;
        handles.Rinf=Rinf;
        handles.Gaussian_fit_x=X;
        handles.Plot_Results=Plot_Results;
        
    end
    
end



guidata(hObject, handles);





% --- Executes on button press in Save_image_dfrt.
function Save_image_dfrt_Callback(hObject, eventdata, handles)
% hObject    handle to Save_image_dfrt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

a=figure('Units', 'Normalized','Position', [0.25, 0.25, 0.35, 0.5]);
SaveImage=copyobj(handles.axes_dfrt, a);
set(SaveImage, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
set(gca,'FontSize',14)
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - 1.2*ti(1) - 1.2*ti(3);
ax_height = outerpos(4) - 1.2*ti(2) - 1.2*ti(4);
ax.Position = [left bottom ax_width ax_height];

guidata(hObject, handles);


function gaussian_terms_Callback(hObject, eventdata, handles)
% hObject    handle to gaussian_terms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gaussian_terms as text
%        str2num(get(hObject,'String')) returns contents of gaussian_terms as a double


% --- Executes during object creation, after setting all properties.
function gaussian_terms_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gaussian_terms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in add_export.
function add_export_Callback(hObject, eventdata, handles)
% hObject    handle to add_export (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

matrix_gaussian_fit=handles.matrix_gaussian_fit+1

X_DFRT_long = evalin('base', 'X_DFRT_long');




if matrix_gaussian_fit==1
    
    MatrixGaussX=[handles.Gaussian_fit_x];
    assignin('base','MatrixGaussX',MatrixGaussX);
    MatrixGaussY=handles.Gaussian_fit;
    assignin('base','MatrixGaussY',MatrixGaussY);
    MatrixGaussXY=[MatrixGaussX MatrixGaussY];
    assignin('base','MatrixGaussXY',MatrixGaussXY)
    assignin('base','GaussResults',handles.Plot_Results)
    handles.MatrixGaussXY=MatrixGaussXY;
    
else
    
    
    MatrixGaussXY = evalin('base', 'MatrixGaussXY');
    MatrixGaussX=handles.Gaussian_fit_x;
    assignin('base','MatrixGaussX',MatrixGaussX);
    MatrixGaussY=handles.Gaussian_fit;
    assignin('base','MatrixGaussY',MatrixGaussY);
    MatrixGaussXY=[MatrixGaussXY MatrixGaussX MatrixGaussY];
    assignin('base','MatrixGaussXY',MatrixGaussXY)
    PlotResults = evalin('base', 'GaussResults');
    PlotResultsMerge =[PlotResults; handles.Plot_Results];
    assignin('base','GaussResults',PlotResultsMerge)
    
    handles.MatrixGaussXY=MatrixGaussXY;
end



handles.matrix_gaussian_fit=matrix_gaussian_fit;

guidata(hObject, handles);


% --- Executes on button press in save_image_nyquist.
function save_image_nyquist_Callback(hObject, eventdata, handles)
% hObject    handle to save_image_nyquist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

a=figure('Units', 'Normalized','Position', [0.25, 0.25, 0.35, 0.5]);
SaveImage=copyobj(handles.axes_nyquist, a);
set(SaveImage, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
set(gca,'FontSize',14)
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - 1.2*ti(1) - 1.2*ti(3);
ax_height = outerpos(4) - 1.2*ti(2) - 1.2*ti(4);
ax.Position = [left bottom ax_width ax_height];

guidata(hObject, handles);

% --- Executes on button press in edited_range.
function edited_range_Callback(hObject, eventdata, handles)
% hObject    handle to edited_range (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of edited_range


% --- Executes on button press in pushbutton_residuals.
function pushbutton_residuals_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_residuals (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

freq=evalin('base','f_exp');
Z_exp_re = evalin('base', 'Z_re_exp');
Z_exp_imag = evalin('base', 'Z_imag_exp');
Z_fit_re = evalin('base', 'Z_re_DFRT');
Z_fit_imag = evalin('base', 'Z_imag_DFRT');





a=size(Z_fit_imag);
b=size(Z_exp_imag);
if(a(1)~=b(1))
    waitfor(errordlg(['Z_DFRT has ' num2str(a(1)) ' points while Z_exp has ' num2str(b(1)) ' points.' sprintf('\n') 'Please calculate Z_DFRT with the same frequency range and length of Z_exp to calculate the residuals.'],'Error - Unable to calculate residuals'));
    return
else
    Residuals_re=((Z_exp_re-Z_fit_re)./((Z_exp_re.^2 + Z_exp_imag.^2).^0.5)).*100;
    Residuals_imag=((Z_exp_imag-Z_fit_imag)./((Z_exp_re.^2 + Z_exp_imag.^2).^0.5)).*100;
    
end


a=figure('Units', 'Normalized','Position', [0.25, 0.25, 0.35, 0.5]);
set(a, 'Units', 'Normalized', 'OuterPosition', [0.2, 0.3, 0.4, 0.35]);
plot(freq,Residuals_re,'ro','LineWidth',0.5,'MarkerSize',7,'MarkerEdgeColor',[0.3010 0.7450 0.9330],'MarkerFaceColor',[0.3010 0.7450 0.9330])
hold on
plot(freq,Residuals_imag,'d','LineWidth',0.5,'MarkerSize',7,'MarkerEdgeColor',[0.8500 0.3250 0.0980],'MarkerFaceColor',[0.8500 0.3250 0.0980])
legend('Real', 'Imag','Location', 'Best')
set(gca, 'xScale', 'log');
grid on
xlabel('Frequency / Hz')
ylabel('Residuals')
handles.residuals=[freq Residuals_re Residuals_imag];
guidata(hObject, handles);


% --- Executes on button press in simu_gauss.
function simu_gauss_Callback(hObject, eventdata, handles)
% hObject    handle to simu_gauss (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

gaussian_terms = str2num(get(handles.gaussian_terms,'String'));
Rinf=evalin('base','Rinf');
X_DFRT_long = evalin('base', 'X_DFRT_long');
Y_DFRT_long = evalin('base', 'Y_DFRT_long');
Z_re_exp = evalin('base', 'Z_re_exp');
Z_imag_exp = evalin('base', 'Z_imag_exp');



X=X_DFRT_long;
Y=Y_DFRT_long;
f_dfrt= evalin('base', 'f_dfrt');
f_exp=evalin('base','f_exp');



if (get(handles.edited_range,'value')==1)       %restricted range=on
    
    if gaussian_terms==1
        
        
        
        lb=ginput(3);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        XX=[log10(X(indexAtMin1))];
        YY=[Y(indexAtMin1)];
        
        diffsx2 = abs(X - lb(2,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        diffsx3 = abs(X - lb(3,1));
        [minDiffx3, indexAtMinx3] = min(diffsx3);
        
        
        if indexAtMinx2<indexAtMinx3
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx3));
            y_gaussian=Y(indexAtMinx2:indexAtMinx3);
        else
            x_gaussian=log10(X(indexAtMinx3:indexAtMinx2));
            y_gaussian=Y(indexAtMinx3:indexAtMinx2);
        end
        
        
        [fitresult, gof] = GaussianFit1(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit_x=X;
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        Y2=Gaussian_fit;
        X2=Gaussian_fit_x;
        ZZ_dfrt=zeros(1,length(f_dfrt));
        
        f=f_dfrt;
        
        for k=1:length(f)
            
            YY2 = Y2./(1+(1i.*2*pi*f(k).*X2));
            ZZ_dfrt(k)=trapz(log(X2),YY2);
        end
        
        
        ZZ_dfrt=ZZ_dfrt';
        ZZ_dfrtf=(Rinf + ZZ_dfrt);
        zz_real = real(ZZ_dfrtf);
        zz_imag = -imag(ZZ_dfrtf);
        
        
        
        prompt = {'Enter the sample thickness','Enter the area of the sample'};
        dlg_title = 'Enter the sample dimensions in metre (SI) to calculate M*';
        width = 90;
        height = 2; % lines in the edit field.
        num_lines = [height, width];
        defaultans = {'1E-3','1E-5'};
        answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
        thickness = str2double(answer{1});
        Area_sample=str2double(answer{2});
        
        
        z_complex = zz_real + 1i.*zz_imag;
        C0=(8.854187817E-12.*Area_sample)./thickness;
        m_complex=(1i.*(2.*pi.*f_dfrt).*C0.*z_complex);
        m_real=real(m_complex);
        m_imag=imag(m_complex);
        Z_exp_complex= Z_re_exp - 1i.*Z_imag_exp;
        m_complex_exp=(1i.*(2.*pi.*f_exp).*C0.*Z_exp_complex);
        
        
        figure('Units', 'Normalized','Position', [0.15, 0.2, 0.55, 0.55]);
        subplot(2,3,1)
        plot(f_exp,real(m_complex_exp),'ro','MarkerSize',7)
        hold on
        plot(f_dfrt,m_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('M^{\prime}')
        grid on
        legend('M^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,2)
        plot(f_exp,imag(m_complex_exp),'ro','MarkerSize',7)
        hold on
        plot(f_dfrt,m_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('M^{\prime\prime}')
        legend('M^{\prime\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,3)
        plot(real(m_complex_exp),imag(m_complex_exp),'ro','MarkerSize',7)
        hold on
        plot(m_real,m_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        Max_x_fit= max(m_real);
        Max_y_fit= max(m_imag);
        Max_x_exp= max(real(m_complex_exp));
        Max_y_exp= max(imag(m_complex_exp));
        if max(m_real)>max(real(m_complex_exp))
            axis ([0 1.1*max(Max_x_fit,Max_y_fit) 0 1.1*max(Max_x_fit,Max_y_fit)])
        elseif max(real(m_complex_exp))>max(m_real)
            axis ([0 1.1*max(Max_x_exp,Max_y_exp) 0 1.1*max(Max_x_exp,Max_y_exp)])
        end
        axis square
        xlabel('M^{\prime}')
        ylabel('M^{\prime\prime}')
        legend('M* exp','Fit', 'Location', 'Best')
        subplot(2,3,4)
        plot(f_exp,Z_re_exp,'ro','MarkerSize',7)
        hold on
        plot(f_dfrt,zz_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('$Z^{\prime}$','Interpreter', 'Latex')
        grid on
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,5)
        plot(f_exp,Z_imag_exp,'ro','MarkerSize',7)
        hold on
        plot(f_dfrt,-zz_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,6)
        plot(Z_re_exp,Z_imag_exp,'ro','MarkerSize',7)
        hold on
        plot(zz_real,-zz_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        if max(zz_real)>max(Z_re_exp)
            axis ([0 1.1*max(max(zz_real),max(-zz_imag)) 0 1.1*max(max(zz_real),max(-zz_imag))])
        else
            axis ([0 1.1*max(max(Z_re_exp),max(Z_imag_exp)) 0 1.1*max(max(Z_re_exp),max(Z_imag_exp))])
        end
        axis square
        xlabel('$Z^{\prime}$','Interpreter', 'Latex')
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z* exp','Fit', 'Location', 'Best')
        
        
        answer = questdlg('Would you like to save the data?', ...
            'Yes', 'No');
        % Handle response
        switch answer
            case 'Yes'
                
                if length(f_exp)~=length(f_dfrt)
                    
                    N=length(f_dfrt)-length(f_exp);
                    t=zeros(N,1);
                    f_exp=[f_exp; t];
                    Z_re_exp=[Z_re_exp; t];
                    Z_imag_exp=[Z_imag_exp; t];
                    M_re_exp=[real(m_complex_exp); t];
                    M_imag_exp=[imag(m_complex_exp); t];
                    
                else
                    
                    
                end
                
                
                Matrix_f = [f_exp f_dfrt M_re_exp M_imag_exp m_real m_imag Z_re_exp Z_imag_exp zz_real -zz_imag];
                [filename, pathname] = uiputfile('.txt');
                fullname = fullfile(pathname,filename);
                fid = fopen(fullname, 'wt');
                fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'F_exp','F_reconstructed','Real(M_exp)','Imag(M_exp)','Real(M_fit)','Imag(M_fit)','Real(Z_exp)','Imag(Z_exp)','Real(Z_fit)','Imag(Z_fit)');  % header
                fclose(fid);
                dlmwrite(fullname,Matrix_f,'delimiter','\t','precision',['%10.',num2str(12),'f'],'-append');
                
                
            case 'No'
                
        end
        
        
        
        
    elseif gaussian_terms==2
        
        lb=ginput(4);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2))];
        YY=[Y(indexAtMin1),Y(indexAtMin2)];
        diffsx1 = abs(X - lb(3,1));
        [minDiffx1, indexAtMinx1] = min(diffsx1);
        diffsx2 = abs(X - lb(4,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        
        if indexAtMinx1<indexAtMinx2
            x_gaussian=log10(X(indexAtMinx1:indexAtMinx2));
            y_gaussian=Y(indexAtMinx1:indexAtMinx2);
        else
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx1));
            y_gaussian=Y(indexAtMinx2:indexAtMinx1);
        end
        
        
        [fitresult, gof] = GaussianFit2(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        Gaussian_fit = Gaussian_fit1 + Gaussian_fit2;
        
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        fn = {'One','Two','Sum of all peaks'};
        [answer,tf] = listdlg('PromptString',{'Which term you want to simulate? (Left to right)',...
            sprintf('\n'),'Select one',''},...
            'SelectionMode','single','Name','Select menu','ListSize',[300,100],'ListString',fn);
        
        switch answer
            case 1
                Y2=Gaussian_fit1;
            case 2
                Y2=Gaussian_fit2;
            case 3
                Y2=Gaussian_fit;
        end
        
        
        X2=X;
        ZZ_dfrt=zeros(1,length(f_dfrt));
        f=f_dfrt;
        
        for k=1:length(f)
            
            YY2 = Y2./(1+(1i.*2*pi*f(k).*X2));
            ZZ_dfrt(k)=trapz(log(X2),YY2);
        end
        
        
        ZZ_dfrt=ZZ_dfrt';
        ZZ_dfrtf=(Rinf + ZZ_dfrt);
        zz_real = real(ZZ_dfrtf);
        zz_imag = -imag(ZZ_dfrtf);
        
        
        
        prompt = {'Enter the sample thickness','Enter the area of the sample'};
        dlg_title = 'Enter the sample dimensions in metre (SI) to calculate M*';
        width = 90;
        height = 2; % lines in the edit field.
        num_lines = [height, width];
        defaultans = {'1E-3','1E-5'};
        answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
        thickness = str2double(answer{1});
        Area_sample=str2double(answer{2});
        
        
        z_complex = zz_real + 1i.*zz_imag;
        C0=(8.854187817E-12.*Area_sample)./thickness;
        m_complex=(1i.*(2.*pi.*f_dfrt).*C0.*z_complex);
        m_real=real(m_complex);
        m_imag=imag(m_complex);
        Z_exp_complex= Z_re_exp - 1i.*Z_imag_exp;
        m_complex_exp=(1i.*(2.*pi.*f_exp).*C0.*Z_exp_complex);
        
        
        figure('Units', 'Normalized','Position', [0.15, 0.2, 0.55, 0.55]);
        subplot(2,3,1)
        plot(f_exp,real(m_complex_exp),'ro','MarkerSize',7)
        hold on
        plot(f_dfrt,m_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('M^{\prime}')
        grid on
        legend('M^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,2)
        plot(f_exp,imag(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(f_dfrt,m_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('M^{\prime\prime}')
        legend('M^{\prime\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,3)
        plot(real(m_complex_exp),imag(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(m_real,m_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        Max_x_fit= max(m_real);
        Max_y_fit= max(m_imag);
        Max_x_exp= max(real(m_complex_exp));
        Max_y_exp= max(imag(m_complex_exp));
        if max(m_real)>max(real(m_complex_exp))
            axis ([0 1.1*max(Max_x_fit,Max_y_fit) 0 1.1*max(Max_x_fit,Max_y_fit)])
        elseif max(real(m_complex_exp))>max(m_real)
            axis ([0 1.1*max(Max_x_exp,Max_y_exp) 0 1.1*max(Max_x_exp,Max_y_exp)])
        end
        axis square
        xlabel('M^{\prime}')
        ylabel('M^{\prime\prime}')
        legend('M* exp','Fit', 'Location', 'Best')
        subplot(2,3,4)
        plot(f_exp,Z_re_exp,'or','MarkerSize',7)
        hold on
        plot(f_dfrt,zz_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('$Z^{\prime}$','Interpreter', 'Latex')
        grid on
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,5)
        plot(f_exp,Z_imag_exp,'or','MarkerSize',7)
        hold on
        plot(f_dfrt,-zz_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,6)
        plot(Z_re_exp,Z_imag_exp,'or','MarkerSize',7)
        hold on
        plot(zz_real,-zz_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        if max(zz_real)>max(Z_re_exp)
            axis ([0 1.1*max(max(zz_real),max(-zz_imag)) 0 1.1*max(max(zz_real),max(-zz_imag))])
        else
            axis ([0 1.1*max(max(Z_re_exp),max(Z_imag_exp)) 0 1.1*max(max(Z_re_exp),max(Z_imag_exp))])
        end
        axis square
        xlabel('$Z^{\prime}$','Interpreter', 'Latex')
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z* exp','Fit', 'Location', 'Best')
        
        
        
        answer = questdlg('Would you like to save the data?', ...
            'Yes', 'No');
        % Handle response
        switch answer
            case 'Yes'
                
                if length(f_exp)~=length(f_dfrt)
                    
                    N=length(f_dfrt)-length(f_exp);
                    t=zeros(N,1);
                    f_exp=[f_exp; t];
                    Z_re_exp=[Z_re_exp; t];
                    Z_imag_exp=[Z_imag_exp; t];
                    M_re_exp=[real(m_complex_exp); t];
                    M_imag_exp=[imag(m_complex_exp); t];
                    
                else
                    
                    
                end
                
                
                Matrix_f = [f_exp f_dfrt M_re_exp M_imag_exp m_real m_imag Z_re_exp Z_imag_exp zz_real -zz_imag];
                [filename, pathname] = uiputfile('.txt');
                fullname = fullfile(pathname,filename);
                fid = fopen(fullname, 'wt');
                fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'F_exp','F_reconstructed','Real(M_exp)','Imag(M_exp)','Real(M_fit)','Imag(M_fit)','Real(Z_exp)','Imag(Z_exp)','Real(Z_fit)','Imag(Z_fit)');  % header
                fclose(fid);
                dlmwrite(fullname,Matrix_f,'delimiter','\t','precision',['%10.',num2str(12),'f'],'-append');
                
                
            case 'No'
                
        end
        
        
    elseif gaussian_terms==3
        
        lb=ginput(5);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        diffs3 = abs(X - lb(3,1));
        [minDiff3, indexAtMin3] = min(diffs3);
        diffsx1 = abs(X - lb(4,1));
        [minDiffx1, indexAtMinx1] = min(diffsx1);
        diffsx2 = abs(X - lb(5,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        
        
        if indexAtMinx1<indexAtMinx2
            x_gaussian=log10(X(indexAtMinx1:indexAtMinx2));
            y_gaussian=Y(indexAtMinx1:indexAtMinx2);
        else
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx1));
            y_gaussian=Y(indexAtMinx2:indexAtMinx1);
        end
        
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2)),log10(X(indexAtMin3))];
        YY=[Y(indexAtMin1),Y(indexAtMin2),Y(indexAtMin3)];
        
        [fitresult, gof] = GaussianFit3(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        Gaussian_fit3=coef(7).*exp(-((log10(X)-coef(8))/coef(9)).^2);
        Gaussian_fit = Gaussian_fit1 + Gaussian_fit2 + Gaussian_fit3;
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        area(X, Gaussian_fit3,'EdgeAlpha',0,'FaceColor',[255/255 180/255 143/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        fn = {'One','Two','Three','One + Two','Sum of all peaks'};
        [answer,tf] = listdlg('PromptString',{'Which term you want to simulate? (Left to right)',...
            sprintf('\n'),'Select one',''},...
            'SelectionMode','single','Name','Select menu','ListSize',[300,100],'ListString',fn);
        
        switch answer
            case 1
                Y2=Gaussian_fit1;
            case 2
                Y2=Gaussian_fit2;
            case 3
                Y2=Gaussian_fit3;
            case 4
                Y2 = Gaussian_fit1 + Gaussian_fit2;
            case 5
                Y2=Gaussian_fit;
        end
        
        X2=X;
        ZZ_dfrt=zeros(1,length(f_dfrt));
        
        f=f_dfrt;
        
        for k=1:length(f)
            
            YY2 = Y2./(1+(1i.*2*pi*f(k).*X2));
            ZZ_dfrt(k)=trapz(log(X2),YY2);
        end
        
        
        ZZ_dfrt=ZZ_dfrt';
        ZZ_dfrtf=(Rinf + ZZ_dfrt);
        zz_real = real(ZZ_dfrtf);
        zz_imag = -imag(ZZ_dfrtf);
        
        
        
        prompt = {'Enter the sample thickness','Enter the area of the sample'};
        dlg_title = 'Enter the sample dimensions in metre (SI) to calculate M*';
        width = 90;
        height = 2; % lines in the edit field.
        num_lines = [height, width];
        defaultans = {'1E-3','1E-5'};
        answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
        thickness = str2double(answer{1});
        Area_sample=str2double(answer{2});
        
        
        z_complex = zz_real + 1i.*zz_imag;
        C0=(8.854187817E-12.*Area_sample)./thickness;
        m_complex=(1i.*(2.*pi.*f_dfrt).*C0.*z_complex);
        m_real=real(m_complex);
        m_imag=imag(m_complex);
        Z_exp_complex= Z_re_exp - 1i.*Z_imag_exp;
        m_complex_exp=(1i.*(2.*pi.*f_exp).*C0.*Z_exp_complex);
        
        
        figure('Units', 'Normalized','Position', [0.15, 0.2, 0.55, 0.55]);
        subplot(2,3,1)
        plot(f_exp,real(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(f_dfrt,m_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('M^{\prime}')
        grid on
        legend('M^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,2)
        plot(f_exp,imag(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(f_dfrt,m_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('M^{\prime\prime}')
        legend('M^{\prime\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,3)
        plot(real(m_complex_exp),imag(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(m_real,m_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        Max_x_fit= max(m_real);
        Max_y_fit= max(m_imag);
        Max_x_exp= max(real(m_complex_exp));
        Max_y_exp= max(imag(m_complex_exp));
        if max(m_real)>max(real(m_complex_exp))
            axis ([0 1.1*max(Max_x_fit,Max_y_fit) 0 1.1*max(Max_x_fit,Max_y_fit)])
        elseif max(real(m_complex_exp))>max(m_real)
            axis ([0 1.1*max(Max_x_exp,Max_y_exp) 0 1.1*max(Max_x_exp,Max_y_exp)])
        end
        axis square
        xlabel('M^{\prime}')
        ylabel('M^{\prime\prime}')
        legend('M* exp','Fit', 'Location', 'Best')
        subplot(2,3,4)
        plot(f_exp,Z_re_exp,'or','MarkerSize',7)
        hold on
        plot(f_dfrt,zz_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('$Z^{\prime}$','Interpreter', 'Latex')
        grid on
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,5)
        plot(f_exp,Z_imag_exp,'or','MarkerSize',7)
        hold on
        plot(f_dfrt,-zz_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,6)
        plot(Z_re_exp,Z_imag_exp,'or','MarkerSize',7)
        hold on
        plot(zz_real,-zz_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        if max(zz_real)>max(Z_re_exp)
            axis ([0 1.1*max(max(zz_real),max(-zz_imag)) 0 1.1*max(max(zz_real),max(-zz_imag))])
        else
            axis ([0 1.1*max(max(Z_re_exp),max(Z_imag_exp)) 0 1.1*max(max(Z_re_exp),max(Z_imag_exp))])
        end
        axis square
        xlabel('$Z^{\prime}$','Interpreter', 'Latex')
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z* exp','Fit', 'Location', 'Best')
        
        
        answer = questdlg('Would you like to save the data?', ...
            'Yes','No');
        % Handle response
        switch answer
            case 'Yes'
                
                if length(f_exp)~=length(f_dfrt)
                    
                    N=length(f_dfrt)-length(f_exp);
                    t=zeros(N,1);
                    f_exp=[f_exp; t];
                    Z_re_exp=[Z_re_exp; t];
                    Z_imag_exp=[Z_imag_exp; t];
                    M_re_exp=[real(m_complex_exp); t];
                    M_imag_exp=[imag(m_complex_exp); t];
                    
                else
                    
                    
                end
                
                
                Matrix_f = [f_exp f_dfrt M_re_exp M_imag_exp m_real m_imag Z_re_exp Z_imag_exp zz_real -zz_imag];
                [filename, pathname] = uiputfile('.txt');
                fullname = fullfile(pathname,filename);
                fid = fopen(fullname, 'wt');
                fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'F_exp','F_reconstructed','Real(M_exp)','Imag(M_exp)','Real(M_fit)','Imag(M_fit)','Real(Z_exp)','Imag(Z_exp)','Real(Z_fit)','Imag(Z_fit)');  % header
                fclose(fid);
                dlmwrite(fullname,Matrix_f,'delimiter','\t','precision',['%10.',num2str(12),'f'],'-append');
                
            case 'No'
                
        end
        
    elseif gaussian_terms==4
        
        
        
        lb=ginput(6);
        diffs1 = abs(X - lb(1,1));
        [minDiff1, indexAtMin1] = min(diffs1);
        
        diffs2 = abs(X - lb(2,1));
        [minDiff2, indexAtMin2] = min(diffs2);
        
        diffs3 = abs(X - lb(3,1));
        [minDiff3, indexAtMin3] = min(diffs3);
        
        diffs4 = abs(X - lb(4,1));
        [minDiff4, indexAtMin4] = min(diffs4);
        diffsx1 = abs(X - lb(5,1));
        [minDiffx1, indexAtMinx1] = min(diffsx1);
        diffsx2 = abs(X - lb(6,1));
        [minDiffx2, indexAtMinx2] = min(diffsx2);
        
        if indexAtMinx1<indexAtMinx2
            x_gaussian=log10(X(indexAtMinx1:indexAtMinx2));
            y_gaussian=Y(indexAtMinx1:indexAtMinx2);
        else
            x_gaussian=log10(X(indexAtMinx2:indexAtMinx1));
            y_gaussian=Y(indexAtMinx2:indexAtMinx1);
        end
        
        XX=[log10(X(indexAtMin1)),log10(X(indexAtMin2)),log10(X(indexAtMin3)),log10(X(indexAtMin4))];
        YY=[Y(indexAtMin1),Y(indexAtMin2),Y(indexAtMin3),Y(indexAtMin3)];
        
        [fitresult, gof] = GaussianFit4(x_gaussian, y_gaussian,YY,XX);
        
        coef=coeffvalues(fitresult);
        Gaussian_fit1=coef(1).*exp(-((log10(X)-coef(2))/coef(3)).^2);
        Gaussian_fit2=coef(4).*exp(-((log10(X)-coef(5))/coef(6)).^2);
        Gaussian_fit3=coef(7).*exp(-((log10(X)-coef(8))/coef(9)).^2);
        Gaussian_fit4=coef(10).*exp(-((log10(X)-coef(11))/coef(12)).^2);
        Gaussian_fit = Gaussian_fit1 + Gaussian_fit2 + Gaussian_fit3 + Gaussian_fit4;
        
        axes(handles.axes_dfrt)
        cla;
        area(X, Gaussian_fit1,'EdgeAlpha',0,'FaceColor',[153/255 255/255 204/255])
        hold on
        area(X, Gaussian_fit2,'EdgeAlpha',0,'FaceColor',[102/255 204/255 255/255])
        area(X, Gaussian_fit3,'EdgeAlpha',0,'FaceColor',[255/255 180/255 143/255])
        area(X, Gaussian_fit4,'EdgeAlpha',0,'FaceColor',[221/255 221/255 221/255])
        plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
        set(gca, 'XScale', 'log');
        xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
        ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
        set(gca, 'XLimSpec', 'Tight');
        
        
        fn = {'One','Two','Three','Four','One + Two','One + Two + Three','Sum of all peaks'};
        [answer,tf] = listdlg('PromptString',{'Which term you want to simulate? (Left to right)',...
            sprintf('\n'),'Select one',''},...
            'SelectionMode','single','Name','Select menu','ListSize',[300,100],'ListString',fn);
        
        switch answer
            case 1
                Y2=Gaussian_fit1;
            case 2
                Y2=Gaussian_fit2;
            case 3
                Y2=Gaussian_fit3;
            case 4
                Y2=Gaussian_fit4;
            case 5
                Y2 = Gaussian_fit1 + Gaussian_fit2;
            case 6
                Y2 = Gaussian_fit1 + Gaussian_fit2 + Gaussian_fit3;
            case 7
                Y2=Gaussian_fit;
        end
        X2=X;
        ZZ_dfrt=zeros(1,length(f_dfrt));
        
        f=f_dfrt;
        
        for k=1:length(f)
            
            YY2 = Y2./(1+(1i.*2*pi*f(k).*X2));
            ZZ_dfrt(k)=trapz(log(X2),YY2);
        end
        
        
        ZZ_dfrt=ZZ_dfrt';
        ZZ_dfrtf=(Rinf + ZZ_dfrt);
        zz_real = real(ZZ_dfrtf);
        zz_imag = -imag(ZZ_dfrtf);
        
        
        
        prompt = {'Enter the sample thickness','Enter the area of the sample'};
        dlg_title = 'Enter the sample dimensions in metre (SI) to calculate M*';
        width = 90;
        height = 2; % lines in the edit field.
        num_lines = [height, width];
        defaultans = {'1E-3','1E-5'};
        answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
        thickness = str2double(answer{1});
        Area_sample=str2double(answer{2});
        
        
        z_complex = zz_real + 1i.*zz_imag;
        C0=(8.854187817E-12.*Area_sample)./thickness;
        m_complex=(1i.*(2.*pi.*f_dfrt).*C0.*z_complex);
        m_real=real(m_complex);
        m_imag=imag(m_complex);
        Z_exp_complex= Z_re_exp - 1i.*Z_imag_exp;
        m_complex_exp=(1i.*(2.*pi.*f_exp).*C0.*Z_exp_complex);
        
        
        figure('Units', 'Normalized','Position', [0.15, 0.2, 0.55, 0.55]);
        subplot(2,3,1)
        plot(f_exp,real(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(f_dfrt,m_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('M^{\prime}')
        grid on
        legend('M^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,2)
        plot(f_exp,imag(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(f_dfrt,m_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('M^{\prime\prime}')
        legend('M^{\prime\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,3)
        plot(real(m_complex_exp),imag(m_complex_exp),'or','MarkerSize',7)
        hold on
        plot(m_real,m_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        Max_x_fit= max(m_real);
        Max_y_fit= max(m_imag);
        Max_x_exp= max(real(m_complex_exp));
        Max_y_exp= max(imag(m_complex_exp));
        if max(m_real)>max(real(m_complex_exp))
            axis ([0 1.1*max(Max_x_fit,Max_y_fit) 0 1.1*max(Max_x_fit,Max_y_fit)])
        elseif max(real(m_complex_exp))>max(m_real)
            axis ([0 1.1*max(Max_x_exp,Max_y_exp) 0 1.1*max(Max_x_exp,Max_y_exp)])
        end
        axis square
        xlabel('M^{\prime}')
        ylabel('M^{\prime\prime}')
        legend('M* exp','Fit', 'Location', 'Best')
        subplot(2,3,4)
        plot(f_exp,Z_re_exp,'or','MarkerSize',7)
        hold on
        plot(f_dfrt,zz_real,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        xlabel('Frequency / Hz');
        ylabel('$Z^{\prime}$','Interpreter', 'Latex')
        grid on
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,5)
        plot(f_exp,Z_imag_exp,'or','MarkerSize',7)
        hold on
        plot(f_dfrt,-zz_imag,'k-','LineWidth',2)
        set(gca, 'xScale', 'log');
        set(gca, 'XLimSpec', 'Tight');
        grid on
        xlabel('Frequency / Hz');
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z^{\prime} exp','Fit', 'Location', 'Best')
        subplot(2,3,6)
        plot(Z_re_exp,Z_imag_exp,'or','MarkerSize',7)
        hold on
        plot(zz_real,-zz_imag,'k-','LineWidth',2)
        set(gca,'XMinorTick','on','YMinorTick','on')
        grid on
        if max(zz_real)>max(Z_re_exp)
            axis ([0 1.1*max(max(zz_real),max(-zz_imag)) 0 1.1*max(max(zz_real),max(-zz_imag))])
        else
            axis ([0 1.1*max(max(Z_re_exp),max(Z_imag_exp)) 0 1.1*max(max(Z_re_exp),max(Z_imag_exp))])
        end
        axis square
        xlabel('$Z^{\prime}$','Interpreter', 'Latex')
        ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
        legend('Z* exp','Fit', 'Location', 'Best')
        
        
        answer = questdlg('Would you like to save the data?', ...
            'Yes', 'No');
        % Handle response
        switch answer
            case 'Yes'
                
                if length(f_exp)~=length(f_dfrt)
                    
                    N=length(f_dfrt)-length(f_exp);
                    t=zeros(N,1);
                    f_exp=[f_exp; t];
                    Z_re_exp=[Z_re_exp; t];
                    Z_imag_exp=[Z_imag_exp; t];
                    M_re_exp=[real(m_complex_exp); t];
                    M_imag_exp=[imag(m_complex_exp); t];
                    
                else
                    
                    
                end
                
                
                Matrix_f = [f_exp f_dfrt M_re_exp M_imag_exp m_real m_imag Z_re_exp Z_imag_exp zz_real -zz_imag];
                [filename, pathname] = uiputfile('.txt');
                fullname = fullfile(pathname,filename);
                fid = fopen(fullname, 'wt');
                fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'F_exp','F_reconstructed','Real(M_exp)','Imag(M_exp)','Real(M_fit)','Imag(M_fit)','Real(Z_exp)','Imag(Z_exp)','Real(Z_fit)','Imag(Z_fit)');  % header
                fclose(fid);
                dlmwrite(fullname,Matrix_f,'delimiter','\t','precision',['%10.',num2str(12),'f'],'-append');
                
            case 'No'
                
        end
        
    end
    
else                    %restricted range off
    
    waitfor(errordlg('Please use the restricted range mode','Error'));
    return
    
end



guidata(hObject, handles);


% --- Executes on button press in simu_cropdata.
function simu_cropdata_Callback(hObject, eventdata, handles)
% hObject    handle to simu_cropdata (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


Rinf=evalin('base','Rinf');
X_DFRT_long = evalin('base', 'X_DFRT_long');
Y_DFRT_long = evalin('base', 'Y_DFRT_long');
Z_re_exp = evalin('base', 'Z_re_exp');
Z_imag_exp = evalin('base', 'Z_imag_exp');


X=X_DFRT_long;
Y=Y_DFRT_long;
f_dfrt= evalin('base', 'f_dfrt');
f_exp=evalin('base','f_exp');

lb=ginput(2);
diffs1 = abs(X - lb(1,1));
[minDiff1, indexAtMinx1] = min(diffs1);
diffsx2 = abs(X - lb(2,1));
[minDiffx2, indexAtMinx2] = min(diffsx2);


if indexAtMinx1<indexAtMinx2
    
    Y2=Y(indexAtMinx1:indexAtMinx2);
    X2=(X(indexAtMinx1:indexAtMinx2));
else
    
    Y2=Y(indexAtMinx2:indexAtMinx1);
    X2=(X(indexAtMinx2:indexAtMinx1));
end



ZZ_dfrt=zeros(1,length(f_dfrt));

f=f_dfrt;

for k=1:length(f)
    
    YY2 = Y2./(1+(1i.*2*pi*f(k).*X2));
    ZZ_dfrt(k)=trapz(log(X2),YY2);
end


ZZ_dfrt=ZZ_dfrt';
ZZ_dfrtf=(Rinf+ ZZ_dfrt);
zz_real = real(ZZ_dfrtf);
zz_imag = -imag(ZZ_dfrtf);



Resistance = trapz(log(X2), Y2);
Plot_Results=[Rinf; max(Z_re_exp); trapz(log(X),Y); Resistance];

CC1={'Rinf','Rtotal (Exp)','Rp (Z_DFRT)','R Gauss1'}';
CC2=num2cell(Plot_Results);
CCC=[CC1 CC2];
set(handles.table_results, 'data', CCC);           %Fill table


prompt = {'Enter the sample thickness','Enter the area of the sample'};
dlg_title = 'Enter the sample dimensions in metre (SI) to calculate M*';
width = 50;
height = 2; % lines in the edit field.
num_lines = [height, width];
defaultans = {'1E-3','1E-5'};
answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
thickness = str2double(answer{1});
Area_sample=str2double(answer{2});



z_complex = zz_real + 1i.*zz_imag;
C0=(8.854187817E-12.*Area_sample)./thickness;
%eps_complex= 1./(1i.*(2.*pi.*f_dfrt).*C0.*z_complex);
m_complex=(1i.*(2.*pi.*f_dfrt).*C0.*z_complex);
m_real=real(m_complex);
m_imag=imag(m_complex);
Z_exp_complex= Z_re_exp - 1i.*Z_imag_exp;
m_complex_exp=(1i.*(2.*pi.*f_exp).*C0.*Z_exp_complex);


figure('Units', 'Normalized','Position', [0.15, 0.2, 0.55, 0.55]);
subplot(2,3,1)
plot(f_exp,real(m_complex_exp),'or','MarkerSize',7)
hold on
plot(f_dfrt,m_real,'k-','LineWidth',2)
set(gca, 'xScale', 'log');
set(gca, 'yScale', 'log');
set(gca, 'XLimSpec', 'Tight');
xlabel('Frequency / Hz');
ylabel('M^{\prime}')
grid on
legend('M^{\prime} exp','Fit', 'Location', 'Best')
subplot(2,3,2)
plot(f_exp,imag(m_complex_exp),'or','MarkerSize',7)
hold on
plot(f_dfrt,m_imag,'k-','LineWidth',2)
set(gca, 'xScale', 'log');
set(gca, 'yScale', 'log');
set(gca, 'XLimSpec', 'Tight');
grid on
xlabel('Frequency / Hz');
ylabel('M^{\prime\prime}')
legend('M^{\prime\prime} exp','Fit', 'Location', 'Best')
subplot(2,3,3)
plot(real(m_complex_exp),imag(m_complex_exp),'or','MarkerSize',7)
hold on
plot(m_real,m_imag,'k-','LineWidth',2)
set(gca,'XMinorTick','on','YMinorTick','on')
grid on
Max_x_fit= max(m_real);
Max_y_fit= max(m_imag);
Max_x_exp= max(real(m_complex_exp));
Max_y_exp= max(imag(m_complex_exp));
if max(m_real)>max(real(m_complex_exp))
    axis ([0 1.1*max(Max_x_fit,Max_y_fit) 0 1.1*max(Max_x_fit,Max_y_fit)])
elseif max(real(m_complex_exp))>max(m_real)
    axis ([0 1.1*max(Max_x_exp,Max_y_exp) 0 1.1*max(Max_x_exp,Max_y_exp)])
end
axis square
xlabel('M^{\prime}')
ylabel('M^{\prime\prime}')
legend('M* exp','Fit', 'Location', 'Best')
subplot(2,3,4)
plot(f_exp,Z_re_exp,'ro','MarkerSize',7)
hold on
plot(f_dfrt,zz_real,'k-','LineWidth',2)
set(gca, 'xScale', 'log');
set(gca, 'XLimSpec', 'Tight');
xlabel('Frequency / Hz');
ylabel('$Z^{\prime}$','Interpreter', 'Latex')
grid on
legend('Z^{\prime} exp','Fit', 'Location', 'Best')
subplot(2,3,5)
plot(f_exp,Z_imag_exp,'ro','MarkerSize',7)
hold on
plot(f_dfrt,-zz_imag,'k-','LineWidth',2)
set(gca, 'xScale', 'log');
set(gca, 'XLimSpec', 'Tight');
grid on
xlabel('Frequency / Hz');
ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
legend('Z^{\prime} exp','Fit', 'Location', 'Best')
subplot(2,3,6)
plot(Z_re_exp,Z_imag_exp,'ro','MarkerSize',7)
hold on
plot(zz_real,-zz_imag,'k-','LineWidth',2)
set(gca,'XMinorTick','on','YMinorTick','on')
grid on
if max(zz_real)>max(Z_re_exp)
    axis ([0 1.1*max(max(zz_real),max(-zz_imag)) 0 1.1*max(max(zz_real),max(-zz_imag))])
else
    axis ([0 1.1*max(max(Z_re_exp),max(Z_imag_exp)) 0 1.1*max(max(Z_re_exp),max(Z_imag_exp))])
end
axis square
xlabel('$Z^{\prime}$','Interpreter', 'Latex')
ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
legend('Z* exp','Fit', 'Location', 'Best')

answer = questdlg('Would you like to save the data?', ...
    'Yes', 'No');
% Handle response
switch answer
    case 'Yes'
        
        if length(f_exp)~=length(f_dfrt)
            
            N=length(f_dfrt)-length(f_exp);
            t=zeros(N,1);
            f_exp=[f_exp; t];
            Z_re_exp=[Z_re_exp; t];
            Z_imag_exp=[Z_imag_exp; t];
            M_re_exp=[real(m_complex_exp); t];
            M_imag_exp=[imag(m_complex_exp); t];
            
        else
            f_exp = f_exp;
            Z_re_exp = Z_re_exp;
            Z_imag_exp = Z_imag_exp;
            M_re_exp = real(m_complex_exp);
            M_imag_exp = imag(m_complex_exp);            
            
        end
        
        
        Matrix_f = [f_exp f_dfrt M_re_exp M_imag_exp m_real m_imag Z_re_exp Z_imag_exp zz_real -zz_imag];
        [filename, pathname] = uiputfile('.txt');
        fullname = fullfile(pathname,filename);
        fid = fopen(fullname, 'wt');
        fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'F_exp','F_reconstructed','Real(M_exp)','Imag(M_exp)','Real(M_fit)','Imag(M_fit)','Real(Z_exp)','Imag(Z_exp)','Real(Z_fit)','Imag(Z_fit)');  % header
        fclose(fid);
        dlmwrite(fullname,Matrix_f,'delimiter','\t','precision',['%10.',num2str(12),'f'],'-append');
        
    case 'No'
        
end


% --- Executes on button press in pushbutton_reset_plot.
function pushbutton_reset_plot_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reset_plot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

X_DFRT_long = evalin('base', 'X_DFRT_long');
Y_DFRT_long = evalin('base', 'Y_DFRT_long');
X=X_DFRT_long;
Y=Y_DFRT_long;
axes(handles.axes_dfrt)
cla;
plot(X,Y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
set(gca, 'XScale', 'log');
xlabel('$\tau /s$', 'Interpreter', 'Latex','Fontsize',14)
ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
set(gca, 'XLimSpec', 'Tight');
