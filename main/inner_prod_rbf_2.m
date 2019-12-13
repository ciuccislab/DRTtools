function out_IP = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type)

a = epsilon*log(freq_n/freq_m);

switch rbf_type
    case 'gaussian'
        out_IP = epsilon^3*(3-6*a^2+a^4)*exp(-(a^2/2))*sqrt(pi/2);
    case 'C0_matern'
        out_IP = epsilon^3*(1+abs(a))*exp(-abs(a));
    case 'C2_matern'
        out_IP = epsilon^3/6*(3+3*abs(a)-6*abs(a)^2+abs(a)^3)*exp(-abs(a));
    case 'C4_matern'
        out_IP = epsilon^3/30*(45+45*abs(a)-15*abs(a)^3-5*abs(a)^4+abs(a)^5)*exp(-abs(a));
    case 'C6_matern'
        out_IP = epsilon^3/140*(2835+2835*abs(a)+630*abs(a)^2-315*abs(a)^3-210*abs(a)^4-42*abs(a)^5+abs(a)^7)*exp(-abs(a));
    case 'inverse_quadratic'
        out_IP = 48*(16+5*a^2*(-8 + a^2))*pi*epsilon^3/((4 + a^2)^5);
    case 'inverse_quadric'
        y_n = -log(freq_n);
        y_m = -log(freq_m);
        % could only find numerical version
        rbf_n = @(y) 1./sqrt(1+(epsilon*(y-y_n)).^2);
        rbf_m = @(y) 1./sqrt(1+(epsilon*(y-y_m)).^2);
        % compute derivative
        delta = 1E-4;
        sqr_drbf_dy = @(y) 1/(delta^2).*(rbf_n(y+delta)-2*rbf_n(y)+rbf_n(y-delta)).*1/(delta^2).*(rbf_m(y+delta)-2*rbf_m(y)+rbf_m(y-delta));
        out_IP = integral(@(y) sqr_drbf_dy(y),-Inf,Inf);       
    case 'cauchy'
        if a == 0
            out_IP = 8/5*epsilon^3;
        else
            num = abs(a)*(2+abs(a))*(-96 +abs(a)*(2+abs(a))*(-30 +abs(a)*(2+abs(a)))*(4+abs(a)*(2+abs(a))))+... 
                    12*(1+abs(a))^2*(16+abs(a)*(2+abs(a))*(12+abs(a)*(2+abs(a))))*log(1+abs(a));
            den = abs(a)^5*(1+abs(a))*(2+abs(a))^5;
            out_IP = 8*epsilon^3*num/den;
        end    
    otherwise
        warning('Unexpected RBF input.');
end

% % begin integrate test:
% y_n = -log(freq_n);
% y_m = -log(freq_m);
% 
% switch rbf_type
%     case 'gaussian'
%         rbf_n = @(y) exp(-(epsilon*(y-y_n)).^2);
%         rbf_m = @(y) exp(-(epsilon*(y-y_m)).^2);
%         
%     case 'C0_matern'
%         rbf_n = @(y)  exp(-abs(epsilon*(y-y_n)));
%         rbf_m = @(y)  exp(-abs(epsilon*(y-y_m)));
%         
%     case 'C2_matern'
%         rbf_n = @(y)  exp(-abs(epsilon*(y-y_n))).*(1+abs(epsilon*(y-y_n)));
%         rbf_m = @(y)  exp(-abs(epsilon*(y-y_m))).*(1+abs(epsilon*(y-y_m)));
%         
%     case 'C4_matern'
%         rbf_n = @(y)  exp(-abs(epsilon*(y-y_n))).*(3+3*abs(epsilon*(y-y_n))+abs(epsilon*(y-y_n)).^2);
%         rbf_m = @(y)  exp(-abs(epsilon*(y-y_m))).*(3+3*abs(epsilon*(y-y_m))+abs(epsilon*(y-y_m)).^2);
%         
%     case 'C6_matern'
%         rbf_n = @(y)  exp(-abs(epsilon*(y-y_n))).*(15+15*abs(epsilon*(y-y_n))+6*abs(epsilon*(y-y_n)).^2+abs(epsilon*(y-y_n)).^3);
%         rbf_m = @(y)  exp(-abs(epsilon*(y-y_m))).*(15+15*abs(epsilon*(y-y_m))+6*abs(epsilon*(y-y_m)).^2+abs(epsilon*(y-y_m)).^3);
%         
%     case 'inverse_quadratic'
%         rbf_n = @(y) 1./(1+(epsilon*(y-y_n)).^2);
%         rbf_m = @(y) 1./(1+(epsilon*(y-y_m)).^2);
%         
%     case 'inverse_quadric'
%         rbf_n = @(y) 1./sqrt(1+(epsilon*(y-y_n)).^2);
%         rbf_m = @(y) 1./sqrt(1+(epsilon*(y-y_m)).^2);
%         
%     case 'cauchy'
%         rbf_n = @(y) 1./(1+abs(epsilon*(y-y_n)));
%         rbf_m = @(y) 1./(1+abs(epsilon*(y-y_m)));
%         
%     otherwise
%         warning('Unexpected RBF input');
%         
% end
% % end of switch
% 
% % compute derivative
% delta = 1E-4;
% sqr_drbf_dy = @(y) 1/(delta^2).*(rbf_n(y+delta)-2*rbf_n(y)+rbf_n(y-delta)).*1/(delta^2).*(rbf_m(y+delta)-2*rbf_m(y)+rbf_m(y-delta));
% % y_dummy = linspace(-10, 10, 10001);
% % plot(y_dummy,sqr_drbf_dy(y_dummy));
% 
% out_IP2 = integral(@(y) sqr_drbf_dy(y),-Inf,Inf);
% 
% fprintf('absolute error = %e \n', abs(out_IP- out_IP2));
% fprintf('rel error = %f percent \n', 100*abs(out_IP- out_IP2)/abs(out_IP));
% if abs(out_IP- out_IP2)/abs(out_IP)>1E-2 pause();end
% % end test

end

