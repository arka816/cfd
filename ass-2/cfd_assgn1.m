
npoints = [10,20,30];
flag = 1;
T_corner = zeros(length(npoints),1);

for p=1:length(npoints)
    %% Declaring variables
    n=npoints(p);
    delta = 1/(n-1);
    Temperature = zeros(n,n);
    A = zeros(n*n,n*n);
    b = zeros(n*n,1);
    corner = zeros(n,1);
        
%% Creating co-efficient matrix and b matrix
    for i=1:n^2
        if i<=n
            A(i,i) = 1;
        elseif i<=n^2-n
            A(i,i) = -4;
            A(i,i-n) = 1;
            A(i,i+n) = 1;
            if(mod(i,n) == 1)
                A(i,i+1) = 2;
            elseif(mod(i,n) == 0)
                A(i,i-1) = 2;
            else
                A(i,i+1) = 1;
                A(i,i-1) = 1;
            end
        else 
            A(i,i) = -4;
            A(i,i-n) = 2;
            if(mod(i,n) == 1)
                A(i,i+1) = 2;
            elseif(mod(i,n) == 0)
                A(i,i-1) = 2;
            else
                A(i,i+1) = 1;
                A(i,i-1) = 1;
            end
        end
    end
    
    for i=1:n 
        b(i,1) = f((i-1)*delta);
    end
    
%% Gauss siedel iteration
%     X = A\b;
    Ab=[A b];
    weight= [1.6, 1.7, 1.80, 1.85, 1.90, 1.95 1.99];
    iteration = zeros(length(weight),1);
    for m=1:length(weight)
        w = weight(m);
        err = 1e-4;
        iter=0;
        X=ones(n*n,1);
        Xold = zeros(n*n,1);
        while max(abs(Xold - X))>err
    %         max(abs(Xold - X))
            iter = iter+1;
            for k=1:n*n
                Xold(k,1) = X(k,1);
                num = Ab(k,end) - Ab(k,1:k-1)*X(1:k-1,1) - Ab(k,k+1:n*n)*X(k+1:n*n,1);
                X(k,1) = Xold(k,1)+w*(num/Ab(k,k)-Xold(k,1));
            end
        end
        iteration(m) = iter;
    end
   
%% Temperature matrix creation
    for i = 1:n
        for j=1:n
            Temperature(i,j) = X((i-1)*n+j, 1);
        end
    end
    Temperature_matrix = zeros(n,n);
    for i = 1:n    
        Temperature_matrix(i,:) = Temperature(n+1-i,:);
    end
    
    
    
%% Plotting
    for i=1:n
        corner(i) = Temperature(i,i);
    end
    xcorner = linspace(0,2^0.5,n);
    T_corner(flag) = T_rms(corner);
    flag = flag+1;
    plot(xcorner,corner);
    hold on;

    x=linspace(0,1,n);
    y=linspace(0,1,n);
    
    pcolor(x,y,Temperature),shading interp,
    title('Temperature(steady state)'),xlabel('x'),ylabel('y'),colorbar;

    plot(weight,iteration,'-o');
    hold on;
end
%  
%     legend('mesh=10*10','mesh=20*20','mesh=30*30','location','east');
%     xlabel('Axis along the diagnonal((0,0) - (1,1))(in metres)');
%     ylabel('Temperature (in degree celcius)');
%     title('grid independence');
%     hold off;

    xlabel('SOR factor');
    ylabel('No. of iterations');
    title('iteration vs weights (to find optimum SOR factor)');
    legend('mesh=10*10','mesh=20*20','mesh=30*30','location','north');
%% functions
function val = f(x)
     val = 100*sin(pi*x);
end

function rms = T_rms(corner)
    n = length(corner);
    res = sum(corner.^2);
    rms = (res/n)^0.5;
end