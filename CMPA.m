
%PA 8: Diode Parameter Extraction

% Part 1
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

V = linspace(-1.95,0.7,200);
I = (Is*(exp(1.2*V/0.025)-1)) + (Gp*V) - (Ib*(exp(-1.2*(V+Vb)/0.025)-1));
r = (1.2-0.8).*rand(size(I)) + 0.8;
Iran =r.*I;

figure()
subplot(2,3,1)
plot(V,I,V,Iran)
legend('I noise', 'I');
xlabel('V');
ylabel('I');
subplot(2,3,2)

semilogy(V,abs(I),V,abs(Iran))
% Part 2
P4 = polyfit(V,I,4);
P8 = polyfit(V,I,8);

V4 = polyval(P4,V);
V8 = polyval(P8,V);

figure(2)
plot(V,I)
hold on
plot(V,V4)
plot(V,V8)
title  'Polynomial Fit'
xlabel ('V');
ylabel ('I');

figure(3)
semilogy(V,abs(I))
hold on
semilogy(V,abs(V4))
semilogy(V,abs(V8))
title  'Polynomial Fit'
xlabel ('V');
ylabel ('I');

% Part 3

%first case
fo1 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff1 = fit(V',I',fo1);
If1 = ff1(V);

%Second Case
fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2 = fit(V',I',fo2);
If2 = ff2(V);

%Third Case
fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3 = fit(V',I',fo3);
If3 = ff3(V);

figure(4)
plot(V,If1)
hold on
plot(V,If2)
plot(V,If3)
legend('If1','If2','If3')

figure(5)
semilogy(V,abs(If1))
%semilogy(V,abs(If1'),'b',V,abs(I20),'r')
hold on
semilogy(V,abs(If2))
semilogy(V,abs(If3))
legend('If1','If2','If3')

% Part 4

inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;

figure(6)
plot(V,Inn)
hold on 
plot(V,I)
title 'Neural Fit'

figure(7)
semilogy(V,abs(Inn))
hold on 
semilogy(V,abs(I))
title 'Neural Fit SemiLog'