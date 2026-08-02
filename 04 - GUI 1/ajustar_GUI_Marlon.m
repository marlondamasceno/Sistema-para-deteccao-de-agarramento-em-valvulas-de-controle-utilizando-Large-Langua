% AJUSTAR A INTERFACE GUI

% 1 - Executar normalmente
iStictionGUI;

% Aguarda a GUI ser criada
pause(1);

mp = get(0,'MonitorPositions');

mon2 = mp(2,:);

% Localiza a janela da GUI
fig = findall(0, ...
    'Type','figure', ...
    'Name','GUI for Oscillation and Stiction Detection in Control Loops');

if isempty(fig)
    error('A janela da iStictionGUI não foi encontrada.');
end

set(fig,'Units','pixels');

set(fig,'Position',...
    [mon2(1)+200  mon2(2)+200  mon2(3)-550  mon2(4)-350]);

% 3 - Posicionar o Painel 3 (Painel verde do teste de agarramento)
painel = findall(fig,'Type','uipanel');

p3 = painel(1);

set(p3,'Visible','on');
set(p3,'Units','characters');

pos = get(p3,'Position');
pos(2) = 15;

set(p3,'Position',pos);
uistack(p3,'top');

% 4 - Ocultar o Quadro Verde dos Resultados para Escolher o método de
% detecção de agarramento

mostrarMetodos = @() set(p3,'Visible','off');
mostrarResultados = @() set(p3,'Visible','on');

% Se desejar mostrar os métodos para selecionar o desejado: mostrarMetodos()
% Se desejar mostrar os resultados: mostrarResultados()

% 5 - Ajustar o quadro verde (para a direita e para baixo)

for x = 3:5:50
    pos = get(p3,'Position');
    pos(1) = x;
    set(p3,'Position',pos);
    pause(0.001)
end

for y = pos(2):-1:0
    pos = get(p3,'Position');
    pos(2) = y;
    set(p3,'Position',pos);
    pause(0.001)
end

% 6 - Mostrar os gráficos ocultos no quadro cinza
painel = findall(fig,'Type','uipanel');
eixos = findall(fig,'Type','axes');
p1 = painel(3);

axSup = eixos(2);
axInf = eixos(3);
axDir = eixos(4);

set([axSup axInf axDir],'Units','pixels');
set(p1,'Units','pixels');

posSupFig = getpixelposition(axSup,true);
posInfFig = getpixelposition(axInf,true);
posDirFig = getpixelposition(axDir,true);

posPainelFig = getpixelposition(p1,true);

set(axSup,'Parent',p1);
set(axInf,'Parent',p1);
set(axDir,'Parent',p1);

posSupPainel = posSupFig;
posInfPainel = posInfFig;
posDirPainel = posDirFig;

posSupPainel(1:2) = posSupFig(1:2) - posPainelFig(1:2);
posInfPainel(1:2) = posInfFig(1:2) - posPainelFig(1:2);
posDirPainel(1:2) = posDirFig(1:2) - posPainelFig(1:2);

set(axSup,'Position',posSupPainel);
set(axInf,'Position',posInfPainel);
set(axDir,'Position',posDirPainel);

drawnow;

graficos = [axSup axInf axDir];

for k = 1:numel(graficos)

    pos = get(graficos(k),'Position');

    pos(3) = pos(3)*0.80;   % 80% da largura
    pos(4) = pos(4)*0.80;   % 80% da altura

    set(graficos(k),'Position',pos);

end

% 7 - Colocar o gráfico de oscilação dentro do painel amarelo

painel = findall(fig,'Type','uipanel');
eixos  = findall(fig,'Type','axes');

p2    = painel(2);   % painel amarelo
axOsc = eixos(1);    % gráfico de oscilação

if ~isgraphics(p2,'uipanel')
    error('O painel amarelo não foi localizado corretamente.');
end

if ~isgraphics(axOsc,'axes')
    error('O eixo de oscilação não foi localizado corretamente.');
end

% Utilizar pixels durante a conversão
set(p2,'Units','pixels');
set(axOsc,'Units','pixels');

% Posições absolutas em relação à figura
posP2Fig = getpixelposition(p2,true);
posAxFig = getpixelposition(axOsc,true);

% Alterar o Parent
set(axOsc,'Parent',p2);

% Converter para coordenadas internas do painel
posAxPainel = posAxFig;
posAxPainel(1:2) = posAxFig(1:2) - posP2Fig(1:2);

% Reposicionar
set(axOsc,'Position',posAxPainel);
set(axOsc,'Visible','on');

drawnow;

pos = get(axOsc,'Position');
pos(1) = pos(1) - 42;
pos(3) = pos(3)*0.95;   % reduz para 90% da largura
pos(4) = pos(4)*0.95;   % reduz para 90% da altura

set(axOsc,'Position',pos);

drawnow;
