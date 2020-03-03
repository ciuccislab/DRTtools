function [q,N] = quantile_alter(X,p,dim,method,weights)
%QUANTILE Quantiles of a sample via various methods.
% 
%   Q = IOSR.STATISTICS.QUANTILE(X,P) returns quantiles of the values in X.
%   P is a scalar or a vector of cumulative probability values.  When X is
%   a vector, Q is the same size as P, and Q(i) contains the P(i)-th
%   quantile.  When X is a matrix, the i-th row of Q contains the P(i)-th
%   quantiles of each column of X.  For N-D arrays,
%   IOSR.STATISTICS.QUANTILE operates along the first non-singleton
%   dimension.
% 
%   Q = IOSR.STATISTICS.QUANTILE(X,P,DIM) calculates quantiles along
%   dimension DIM.  The DIM'th dimension of Q has length LENGTH(P).
% 
%   Q = IOSR.STATISTICS.QUANTILE(X,P,DIM,METHOD) calculates quantiles using
%   one of the methods described in http://en.wikipedia.org/wiki/Quantile.
%   The method are designated 'R-1'...'R-9'; the default is R-8 as
%   described in http://bit.ly/1kX4NcT, whereas Matlab uses 'R-5'.
%   
%   Q = IOSR.STATISTICS.QUANTILE(X,P,[],METHOD) uses the specified METHOD,
%   but calculates quantiles along the first non-singleton dimension.
% 
%   Q = IOSR.STATISTICS.QUANTILE(X,P,[],METHOD,WEIGHTS) and
%   IOSR.STATISTICS.QUANTILE(X,P,[],[],WEIGHTS) uses the array WEIGHTS to
%   weight the values in X when calculating quantiles. If no weighting is
%   specified, the method determines the real-valued index in to the data
%   that is used to calculate the P(i)-th quantile. When a weighting array
%   WEIGHTS is specified (WEIGHTS should be the same size as X), this index
%   is mapped to the cumulative weights (the weights are scaled to sum to
%   N(i) - see below), and a new weighted index is returned (using linear
%   interpolation) for the point where the cumulative weights equal the
%   unweighted index. The weighted index is used to calculate the P(i)-th
%   quantile. If the values in WEIGHTS are equal, then the weighted and
%   unweighted index (and correpsonding quantile) are identical. The
%   default method R-8 is used if METHOD is specified as an empty array
%   ([]).
% 
%   [Q,N] = IOSR.STATISTICS.QUANTILE(...) returns an array that is the same
%   size as Q such that N(i) is the number of points used to calculate
%   Q(i).
% 
%   Further reading
%   
%   Hyndman, R.J.; Fan, Y. (November 1996). "Sample Quantiles in
%     Statistical Packages". The American Statistician 50 (4): 361-365.
%   Frigge, Michael; Hoaglin, David C.; Iglewicz, Boris (February 1989).
%     "Some Implementations of the Boxplot". The American Statistician 43
%     (1): 50-54.
%   
%   See also QUANTILE.

%   Copyright 2016 University of Surrey.

    %% Check input and make default assignments

    assert(isnumeric(X), 'iosr:quantile:invalidX', 'X must be a numeric');
    assert(isvector(p) & isnumeric(p), 'iosr:quantile:invalidP', 'P must be a numeric vector');
    assert(all(p>=0 & p<=1), 'iosr:quantile:invalidP', 'Values in P must be in the interval [0,1].')

    if nargin<2
        error('iosr:quantile:tooFewInputArgs','Not enough input arguments.')
    end

    dims = size(X);
    if nargin<3 || isempty(dim)
        dim = find(dims>1,1,'first'); % default dim
    else % validate input
        assert(isnumeric(dim) | isempty(dim), 'iosr:quantile:invalidDim', 'DIM must be an integer or empty');
        assert(isint(dim) | isempty(dim), 'iosr:quantile:invalidDim', 'DIM must be an integer or empty');
        assert(dim>0, 'iosr:quantile:invalidDim', 'DIM must be greater than 0')
    end

    if nargin<4
        method = 'r-8'; % default method
    else % validate input
        if isempty(method)
            method = 'r-8'; % default method
        else
            assert(ischar(method), 'iosr:quantile:invalidMethod', 'METHOD must be a character array')
        end
    end
    
    if nargin<5
        weights = [];
    else
        assert(isequal(size(X),size(weights)) || isempty(weights), 'iosr:quantile:invalidWeights', 'WEIGHTS must be the same size as X');
    end

    %% choose method

    % See http://en.wikipedia.org/wiki/Quantile#Estimating_the_quantiles_of_a_population

    switch lower(method)
        case 'r-1'
            min_con = @(N,p)(p==0);
            max_con = @(N,p)(false);
            h = @(N,p)((N*p)+.5);
            Qp = @(x,h)(x(ceil(h-.5)));
        case 'r-2'
            min_con = @(N,p)(p==0);
            max_con = @(N,p)(p==1);
            h = @(N,p)((N*p)+.5);
            Qp = @(x,h)((x(ceil(h-.5))+x(floor(h+.5)))/2);
        case 'r-3'
            min_con = @(N,p)(p<=(.5/N));
            max_con = @(N,p)(false);
            h = @(N,p)(N*p);
            Qp = @(x,h)(x(round(h)));
        case 'r-4'
            min_con = @(N,p)(p<(1/N));
            max_con = @(N,p)(p==1);
            h = @(N,p)(N*p);
            Qp = @(x,h)(x(floor(h)) + ((h-floor(h))*(x(floor(h)+1)-x(floor(h)))));
        case 'r-5'
            min_con = @(N,p)(p<(.5/N));
            max_con = @(N,p)(p>=((N-.5)/N));
            h = @(N,p)((N*p)+.5);
            Qp = @(x,h)(x(floor(h)) + ((h-floor(h))*(x(floor(h)+1)-x(floor(h)))));
        case 'r-6'
            min_con = @(N,p)(p<(1/(N+1)));
            max_con = @(N,p)(p>=(N/(N+1)));
            h = @(N,p)((N+1)*p);
            Qp = @(x,h)(x(floor(h)) + ((h-floor(h))*(x(floor(h)+1)-x(floor(h)))));
        case 'r-7'
            min_con = @(N,p)(false);
            max_con = @(N,p)(p==1);
            h = @(N,p)(((N-1)*p)+1);
            Qp = @(x,h)(x(floor(h)) + ((h-floor(h))*(x(floor(h)+1)-x(floor(h)))));
        case 'r-8'
            min_con = @(N,p)(p<((2/3)/(N+(1/3))));
            max_con = @(N,p)(p>=((N-(1/3))/(N+(1/3))));
            h = @(N,p)(((N+(1/3))*p)+(1/3));
            Qp = @(x,h)(x(floor(h)) + ((h-floor(h))*(x(floor(h)+1)-x(floor(h)))));
        case 'r-9'
            min_con = @(N,p)(p<((5/8)/(N+.25)));
            max_con = @(N,p)(p>=((N-(3/8))/(N+.25)));
            h = @(N,p)(((N+.25)*p)+(3/8));
            Qp = @(x,h)(x(floor(h)) + ((h-floor(h))*(x(floor(h)+1)-x(floor(h)))));
        otherwise
            error('iosr:quantile:unknownMethod',['Method ''' method ''' does not exist'])
    end

    %% calculate quartiles

    % reshape data so function works down columns
    order = mod(dim-1:dim+length(dims)-2,length(dims))+1;
    dims_shift = dims(order);
    x = rearrange(X,order,[dims_shift(1) prod(dims_shift(2:end))]);
    if ~isempty(weights)
        weights = rearrange(weights,order,[dims_shift(1) prod(dims_shift(2:end))]);
        cumwfunc = @accumulateWeights;
        wfunc = @weightedIndex;
    else
        cumwfunc = @(~,~,~,N) 1:N;
        wfunc = @(x,~) x;
    end

    % pre-allocate q
    q = zeros([length(p) prod(dims_shift(2:end))]);
    N = zeros([length(p) prod(dims_shift(2:end))]);
    for m = 1:length(p)
        for n = 1:numel(q)/length(p)
            [xSorted,ind] = sort(x(~isnan(x(:,n)),n)); % sort
            N(m,n) = length(xSorted); % sample size
            k = cumwfunc(weights,ind,n,N(m,n));
            switch N(m,n)
                case 0
                    q(m,n) = NaN;
                case 1
                    q(m,n) = xSorted;
                otherwise
                    if min_con(N(m,n),p(m)) % at lower limit
                        q(m,n) = xSorted(1);
                    elseif max_con(N(m,n),p(m)) % at upper limit
                        q(m,n) = xSorted(N(m,n));
                    else % everything else
                        huw = h(N(m,n),p(m)); % unweighted index
                        hw = wfunc(huw,k);
                        q(m,n) = Qp(xSorted,hw);
                    end
            end
        end
    end

    % restore dims of q to equate to those of input
    q = irearrange(q,order,[length(p) dims_shift(2:end)]);
    N = irearrange(N,order,[length(p) dims_shift(2:end)]);

    % if q is a vector, make same shape as p
    if numel(p)==numel(q)
        q=reshape(q,size(p));
        N=reshape(N,size(p));
    end

end

function cumweights = accumulateWeights(weights,ind,n,N)
%ACCUMULATEWEIGHTS accumulate the weights

    wSorted = weights(ind,n); % sort weights
    wSorted = wSorted*N/sum(wSorted); % normalize weights to sum to N
    cumweights = cumsum(wSorted); % cumulative weights

end

function hw = weightedIndex(huw, cumweights)
%WEIGHTEDINDEX calculate index from cumulative weights

    ii = find(sign(cumweights-huw)<0,1,'last');
    jj = find(sign(cumweights-huw)>0,1,'first');
    if isempty(ii) || isempty(jj)
        hw = huw;
    else
        hw = ii + (huw-cumweights(ii))/(cumweights(jj)-cumweights(ii)); % weighted index
    end
    
end

function y = isint(x)
%ISINT check if input is whole number
    y = x==round(x);
end

function y = rearrange(x,order,shape)
%REARRANGE reshape and permute to make target dim column
    y = permute(x,order);
    y = reshape(y,shape);
end

function y = irearrange(x,order,shape)
%IREARRANGE reshape and permute to original size
    y = reshape(x,shape);
    y = ipermute(y,order);
end
