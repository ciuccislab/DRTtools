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

function varargout = DRTtoEIS_main(varargin)
% Begin initialization code - DO NOT EDIT

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DFRTtoEIS_main_OpeningFcn, ...
                   'gui_OutputFcn',  @DFRTtoEIS_main_OutputFcn, ...
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


% --- Outputs from this function are returned to the command line.
function varargout = DFRTtoEIS_main_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;


% --- Executes just before DFRTtoEIS_main is made visible.
function DFRTtoEIS_main_OpeningFcn(hObject, eventdata, handles, varargin)

% if ~license('test', 'Optimization_Toolbox')
% 
%     error('***Optimization Toolbox licence is missing, DRTtools terminated***')
%     close(DRTtools)
%     
% end
% Choose default command line output for DFRTtoEIS_main
handles.output = hObject;


% Update handles structure

guidata(hObject, handles);






% --- Executes on selection change in LegendStyle.
function popupmenu_dfrt_Callback(hObject, eventdata, handles)
% hObject    handle to LegendStyle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns LegendStyle contents as cell array
%        contents{get(hObject,'Value')} returns selected item from LegendStyle


% --- Executes during object creation, after setting all properties.
function popupmenu_dfrt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LegendStyle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end 
 
 
 function results_Callback(hObject, eventdata, handles)
% hObject    handle to results (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of results as text
%        str2num(get(hObject,'String')) returns contents of results as a double


% --- Executes during object creation, after setting all properties.
function results_CreateFcn(hObject, eventdata, handles)
% hObject    handle to results (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on selection change in Yscale_real.
function Yreal_Callback(hObject, eventdata, handles)
% hObject    handle to Yscale_real (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Yscale_real contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Yscale_real


% --- Executes during object creation, after setting all properties.
function Yreal_real_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Yscale_real (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in Yscale_real.
function Yimag_Callback(hObject, eventdata, handles)
% hObject    handle to Yscale_real (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Yscale_real contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Yscale_real


% --- Executes during object creation, after setting all properties.
function Yimag_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Yscale_real (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_residuals.
function pushbutton_residuals_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_residuals (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

freq=handles.freq;
Z_exp_re = handles.Zreal;
Z_exp_imag = -handles.Zimag;
Z_fit_re = handles.Z_dfrtf_re;
Z_fit_imag = -handles.Z_dfrtf_imag;



a=length(Z_fit_imag);
b=length(Z_exp_imag);
if(a~=b)

    waitfor(errordlg(['Z_DFRT has ' num2str(a(1)) ' points while Z_exp has ' num2str(b) ' points.' sprintf('\n') 'Please calculate Z_DFRT with the same frequency range and length of Z_exp to calculate the residuals.'],'Error - Unable to calculate residuals'));
    return
else
  Residuals_re=((Z_exp_re-Z_fit_re)./((Z_exp_re.^2 + Z_exp_imag.^2).^0.5)).*100;
  Residuals_imag=((Z_exp_imag-Z_fit_imag)./((Z_exp_re.^2 + Z_exp_imag.^2).^0.5)).*100;

end    


a=figure('Units', 'Normalized','Position', [0.25, 0.25, 0.35, 0.5]);
set(a, 'Units', 'Normalized', 'OuterPosition', [0.2, 0.3, 0.4, 0.35]);
plot(freq,Residuals_re,'o','LineWidth',0.5,'MarkerSize',7,'MarkerEdgeColor',[0.3010 0.7450 0.9330],'MarkerFaceColor',[0.3010 0.7450 0.9330])
hold on
plot(freq,Residuals_imag,'d','LineWidth',0.5,'MarkerSize',7,'MarkerEdgeColor',[0.8500 0.3250 0.0980],'MarkerFaceColor',[0.8500 0.3250 0.0980])
legend('Real', 'Imag','Location', 'Best')
set(gca, 'xScale', 'log');
grid on
xlabel('$f$/Hz', 'Interpreter', 'Latex')
ylabel('$Residuals / {\%} $', 'Interpreter', 'Latex')
set(gca,'FontSize',13)    
handles.residuals=[freq Residuals_re Residuals_imag];

figure
ax1=subplot(1,2,1); % Left subplot
histfit(Residuals_re)
title(ax1,'Real')
ax2=subplot(1,2,2); % Right subplot
histfit(Residuals_imag)
title(ax2,'Imaginary')

guidata(hObject, handles);


% --- Executes on button press in pushbutton_goback.
function pushbutton_goback_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_goback (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

X = getappdata(0, 'tau');               %load extended range DFRT
X=X.*2.*pi;
Y = getappdata(0, 'DFRT');               %load exteneded range DFRT
handles.freq = getappdata(0, 'freq');
handles.Zreal = getappdata(0, 'Zreal');
handles.Zimag = getappdata(0, 'Zimag');
handles.Rinf = getappdata(0, 'Rinf');
handles.L = getappdata(0, 'L');


f=1./(2.*pi.*X);


n = get(handles.popupmenu_dfrt,'value');
contentFreqImag_r = get(handles.Yreal,'String');
popupmenuFreqImag_r = contentFreqImag_r{get(handles.Yreal,'Value')};
contentFreqImag_i = get(handles.Yimag,'String');
popupmenuFreqImag_i = contentFreqImag_i{get(handles.Yimag,'Value')};

if n==1         %default range
    
    
    
    resistance=[max(handles.Zreal); trapz(log(X),Y); handles.Rinf; handles.L];
    set(handles.results,'string',num2str(resistance,'%.4G'));   
    
    Z_dfrt=zeros(1,length(f));
    
    for k=1:length(f)
        
        YY = Y./(1+(1i.*2*pi*f(k).*X));
        assignin('base','f',handles.freq)
        Z_dfrt(k)=trapz(log(X),YY);
    end
    

    Z_dfrtf=(handles.Rinf + Z_dfrt)';
    Z_dfrtf_re = real(Z_dfrtf); 
    Z_dfrtf_imag = -imag(Z_dfrtf) + 2.*pi.*f.*handles.L;
    
    Maxx= max(Z_dfrtf_re);
    Maxy= max(-Z_dfrtf_imag);
    Maxx2= max(handles.Zreal);
    Maxy2= max(-handles.Zimag);
    Maxf=max(Maxx,Maxx2);
    Maxyf=max(Maxy,Maxy2);
    
    axes(handles.axes_goback)
    cla;
    plot(Z_dfrtf_re,-Z_dfrtf_imag,'k-','LineWidth',2)
    hold on
    plot(handles.Zreal,-handles.Zimag,'or', 'MarkerSize', 6, 'MarkerFaceColor', 'r')    
    set(gca,'FontSize',10)
    set(gca, 'YScale', 'linear');
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on
    axis ([0 1.1*max(Maxf,Maxyf) 0 1.1*max(Maxf,Maxyf)])
    xlabel('$Z^{\prime}$','Interpreter', 'Latex')
    ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')    
    axis square
    grid on
    box on
    axis equal
    pbaspect([1 1 1])
    hold off
    
      
      
    axes(handles.axes_dfrtr)
    cla;
    plot(f,Z_dfrtf_re,'k-','LineWidth',4)
    hold on
    plot(handles.freq, handles.Zreal,'or', 'MarkerSize', 2, 'MarkerFaceColor', 'r')
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', popupmenuFreqImag_r);        
    xlabel('$f$/Hz', 'Interpreter', 'Latex')
    ylabel('$Z^{\prime}$','Interpreter', 'Latex')
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on
    set(gca,'FontSize',11)    
    axis tight    
    
    axes(handles.axes_dfrtimag)
    cla;
    plot(f,-Z_dfrtf_imag,'k-','LineWidth',4)
    hold on
    plot(handles.freq, -handles.Zimag,'or', 'MarkerSize', 2, 'MarkerFaceColor', 'r')
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', popupmenuFreqImag_i);        
    xlabel('$f$/Hz', 'Interpreter', 'Latex')
    ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on    
    set(gca,'FontSize',11)      
    axis tight

elseif n==2

           

    prompt = {'Enter low frequency as log10(value)','Enter high frequency as log10(value)','Points per decade of frequency'};
    dlg_title = 'Input of the frequency range';
    width = 50;
    height = 2; % lines in the edit field.
    num_lines = [height, width];
    defaultans = {'-2','8','10'};
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    fstart = str2double(answer{1});
    fend = str2double(answer{2});
    f_decades = str2double(answer{3});
    if fstart<0
        
        f_decades2=fend+abs(fstart);
    else
        f_decades2=fend-abs(fstart);
    end
    
    f = logspace(fstart,fend,(f_decades*f_decades2+1))';
    
    
    resistance=[max(handles.Zreal); trapz(log(X),Y); handles.Rinf; handles.L];
    set(handles.results,'string',num2str(resistance,'%.4G'));    
    Z_dfrt=zeros(1,length(f));
    
    for k=1:length(f)
        
        YY = Y./(1+(1i.*2*pi*f(k).*X));
        Z_dfrt(k)=trapz(log(X),YY);
    end
    
    
    
    Z_dfrtf=(handles.Rinf + Z_dfrt)';
    Z_dfrtf_re = real(Z_dfrtf); 
    Z_dfrtf_imag = -imag(Z_dfrtf) + 2.*pi.*f.*handles.L;
    
    Maxx= max(Z_dfrtf_re);
    Maxy= max(-Z_dfrtf_imag);
    Maxx2= max(handles.Zreal);
    Maxy2= max(-handles.Zimag);
    Maxf=max(Maxx,Maxx2);
    Maxyf=max(Maxy,Maxy2);
    
    axes(handles.axes_goback)
    cla;
    plot(Z_dfrtf_re,-Z_dfrtf_imag,'k-','LineWidth',2)
    hold on
    plot(handles.Zreal,-handles.Zimag,'or', 'MarkerSize', 6, 'MarkerFaceColor', 'r')    
    set(gca,'FontSize',10)
    set(gca, 'YScale', 'linear');
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on
    axis ([0 1.1*max(Maxf,Maxyf) 0 1.1*max(Maxf,Maxyf)])
    xlabel('$Z^{\prime}$','Interpreter', 'Latex')
    ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')    
    axis square
    grid on
    box on
    axis equal
    pbaspect([1 1 1])
    hold off
    
    
    
    
      
    axes(handles.axes_dfrtr)
    cla;
    plot(f,Z_dfrtf_re,'k-','LineWidth',4)
    hold on
    plot(handles.freq, handles.Zreal,'or', 'MarkerSize', 2, 'MarkerFaceColor', 'r')
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', popupmenuFreqImag_r);        
    axis tight
    xlabel('$f$/Hz', 'Interpreter', 'Latex')
    ylabel('$Z^{\prime}$','Interpreter', 'Latex')
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on
    set(gca,'FontSize',11)    
    
    axes(handles.axes_dfrtimag)
    cla;
    plot(f,-Z_dfrtf_imag,'k-','LineWidth',4)
    hold on
    plot(handles.freq, -handles.Zimag,'or', 'MarkerSize', 2, 'MarkerFaceColor', 'r')
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', popupmenuFreqImag_i);        
    axis tight
    xlabel('$f$/Hz', 'Interpreter', 'Latex')
    ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on    
    set(gca,'FontSize',11)  

elseif n==3
    
    f=handles.freq;
    resistance=[max(handles.Zreal); trapz(log(X),Y); handles.Rinf; handles.L];
    set(handles.results,'string',num2str(resistance,'%.4G'));    
    Z_dfrt=zeros(1,length(f));
    
    for k=1:length(f)
        
        YY = Y./(1+(1i.*2*pi*f(k).*X));
        Z_dfrt(k)=trapz(log(X),YY);
    end
    
    
   
    Z_dfrtf=(handles.Rinf + Z_dfrt)';
    Z_dfrtf_re = real(Z_dfrtf); 
    Z_dfrtf_imag = -imag(Z_dfrtf) + 2.*pi.*f.*handles.L;
    
    Maxx= max(Z_dfrtf_re);
    Maxy= max(-Z_dfrtf_imag);
    Maxx2= max(handles.Zreal);
    Maxy2= max(-handles.Zimag);
    Maxf=max(Maxx,Maxx2);
    Maxyf=max(Maxy,Maxy2);
    
    axes(handles.axes_goback)
    cla;
    plot(Z_dfrtf_re,-Z_dfrtf_imag,'k-','LineWidth',2)
    hold on
    plot(handles.Zreal,-handles.Zimag,'or', 'MarkerSize', 3, 'MarkerFaceColor', 'r')    
    set(gca,'FontSize',10)
    set(gca, 'YScale', 'linear');
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on
    if max(Maxf,Maxyf)>1        
        axis ([0 1.1*max(Maxf,Maxyf) 0 1.1*max(Maxf,Maxyf)])
    end
    xlabel('$Z^{\prime}$','Interpreter', 'Latex')
    ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')    
    grid on
    box on
    axis equal
    pbaspect([1 1 1])   
    hold off
    
    
    
    
      
    axes(handles.axes_dfrtr)
    cla;
    plot(f,Z_dfrtf_re,'k-','LineWidth',4)
    hold on
    plot(handles.freq, handles.Zreal,'or', 'MarkerSize', 2, 'MarkerFaceColor', 'r')
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', popupmenuFreqImag_r);        
    axis tight
    xlabel('$f$/Hz', 'Interpreter', 'Latex')
    ylabel('$Z^{\prime}$','Interpreter', 'Latex')
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on
    set(gca,'FontSize',11)    
    
    axes(handles.axes_dfrtimag)
    cla;
    plot(f,-Z_dfrtf_imag,'k-','LineWidth',4)
    hold on
    plot(handles.freq, -handles.Zimag,'or', 'MarkerSize', 2, 'MarkerFaceColor', 'r')
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', popupmenuFreqImag_i);        
    axis tight
    xlabel('$f$/Hz', 'Interpreter', 'Latex')
    ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
    set(gca,'XMinorTick','on','YMinorTick','on')
    grid on    
    set(gca,'FontSize',11)  
    
end

handles.f=f;
handles.Z_dfrtf_re = Z_dfrtf_re;
handles.Z_dfrtf_imag = Z_dfrtf_imag;
handles.X = X;
handles.Y = Y;
guidata(hObject,handles)

% --- Executes on button press in goback_nw.
function goback_nw_Callback(hObject, eventdata, handles)
% hObject    handle to goback_nw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


x=handles.X;               %load extended range DFRT
y=handles.Y;               %load exteneded range DFRT
run('DFRTtoEIS_nw')
DFRT_minee=guidata(DFRTtoEIS_nw);


axes(DFRT_minee.axes_dfrt)

plot(x,y,'-', 'Color', [82 82 82]/255, 'LineWidth', 3);
set(gca, 'XScale', 'log');
xlabel('$\tau \ /s$', 'Interpreter', 'Latex','Fontsize',14)
ylabel('$Rp.G(\tau) / \Omega$','Interpreter', 'Latex','Fontsize',14);
set(gca, 'XLimSpec', 'Tight');




axes(handles.axes_goback);
h = findobj(gca,'Type','line');
x2=get(h,'Xdata');
y2=get(h,'Ydata');


Maxx= max(x2{1,:});
Maxy= max(y2{1,:});
Maxx2= max(x2{2,:});
Maxy2= max(y2{2,:});
Maxf=max(Maxx,Maxx2);
Maxyf=max(Maxy,Maxy2);


axes(DFRT_minee.axes_nyquist)
cla;
plot(x2{2,:},y2{2,:},'k-','LineWidth',2)
hold on 
plot(x2{1,:},y2{1,:},'or', 'MarkerSize', 3, 'MarkerFaceColor', 'r')
set(gca,'FontSize',13)
set(gca, 'YScale', 'linear');
set(gca, 'XScale', 'linear');
set(gca,'XMinorTick','on','YMinorTick','on')
xlabel('$Z^{\prime}$','Interpreter', 'Latex')
ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
if max(Maxf,Maxyf)>1
    axis ([0 1.1*max(Maxf,Maxyf) 0 1.1*max(Maxf,Maxyf)])
end
xlabel('$Z^{\prime}$','Interpreter', 'Latex')
ylabel('$-Z^{\prime\prime}$','Interpreter', 'Latex')
grid on
box on
axis equal
pbaspect([1 1 1])
hold off

assignin('base','X_DFRT_long',x);      %save DRT
assignin('base','Y_DFRT_long',y);      %save DRT 
assignin('base','Z_re_exp',x2{1,:}');      %save Z' Exp
assignin('base','Z_imag_exp',y2{1,:}');      %save Z'' exp
assignin('base','Z_re_DFRT',x2{2,:}');     %save Z' from DFRT
assignin('base','Z_imag_DFRT',y2{2,:}');     %save Z'' from DFRT
assignin('base','Rinf',handles.Rinf);      %save Rinf
assignin('base','f_exp',handles.freq);
assignin('base','f_dfrt',handles.f);

guidata(hObject,handles)




% --- Executes during object creation, after setting all properties.
function Yreal_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Yreal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function Untitled_1_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
