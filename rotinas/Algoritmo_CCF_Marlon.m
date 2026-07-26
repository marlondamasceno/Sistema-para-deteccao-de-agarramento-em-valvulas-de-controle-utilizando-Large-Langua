%% ALGORITMO DE FUNCAO DE CORRELACAO CRUZADA (CCF)
% Avalia a CCF entre os sinais MV e PV de um arquivo CSV.
%
% Critério de tolerância Delta_phi:
%   Tp = 1/fp
%   Delta_phi = Tp/12
%
% Classificacao adotada:
%
%   Delta_tau e Delta_rho <= 1/3  -> CCF aproximadamente par
%   Delta_tau e Delta_rho >= 2/3  -> CCF aproximadamente impar
%   Demais casos                  -> zona morta (paridade indefinida)

clear;
clc;
close all;

%% ETAPA 1 - INSERIR O ARQUIVO DE AMOSTRA

%% 1.1 Importar o arquivo CSV
arquivo = '2026.04.07 - 312-RV-401-LIC-02.csv';
opts = detectImportOptions(arquivo, 'Delimiter', ',');
opts.VariableNamingRule = 'preserve';
dados = readtable(arquivo, opts);

%% 1.2 Verificar a integridade dos dados importados

colunasNecessarias = {'tempo', 'PV', 'MV'};

for i = 1:numel(colunasNecessarias)
    nomeColuna = colunasNecessarias{i};
    if ~ismember(nomeColuna, dados.Properties.VariableNames)
        error('A coluna "%s" nao foi encontrada no arquivo.', ...
              nomeColuna);
    end
end

% Remover linhas sem tempo, PV ou MV
dados = rmmissing( ...
    dados, ...
    'DataVariables', colunasNecessarias);

% Ordenar os dados pelo tempo
dados = sortrows(dados, 'tempo');

% Tratar instantes de tempo repetidos por meio da media dos valores
[tempoUnico, ~, grupo] = unique(dados.tempo, 'sorted');
PVmedio = accumarray(grupo, dados.PV, [], @mean);
MVmedio = accumarray(grupo, dados.MV, [], @mean);

% Verificar se existe sinal de SP
if ismember('SP', dados.Properties.VariableNames)
    SPmedio = accumarray(grupo, dados.SP, [], @mean);
else
    SPmedio = NaN(size(tempoUnico));
end

dados = table(tempoUnico, SPmedio, PVmedio, MVmedio, 'VariableNames', ...
    {'tempo', 'SP', 'PV', 'MV'});

%% 1.3 Selecionar o intervalo de dados que será analisado

% Este item é opcional, pois dentro do arquivo de amostras há trecho em que
% ocorrem degraus e outros eventos em que não corroboram com os requisitos
% da CCF

% Use NaN para analisar todo o arquivo.
% tInicio = NaN;
% tFim    = NaN;
tInicio = 11500;
tFim    = 12500;

if ~isnan(tInicio)
    dados = dados(dados.tempo >= tInicio, :);
end

if ~isnan(tFim)
    dados = dados(dados.tempo <= tFim, :);
end

if height(dados) < 10
    error(['O intervalo selecionado possui poucas amostras ', ...
           'para realizar a analise.']);
end

% Verificar e uniformizar o periodo de amostragem

dt = diff(dados.tempo);
dtValidos = dt(dt > 0 & isfinite(dt));

if isempty(dtValidos)
    error('Nao foi possivel determinar o periodo de amostragem.');
end

Ts = median(dtValidos);

if ~isfinite(Ts) || Ts <= 0
    error('O periodo de amostragem calculado e invalido.');
end

tempoUniforme = (dados.tempo(1):Ts:dados.tempo(end))';

PV = interp1(dados.tempo, dados.PV, tempoUniforme, 'linear');
MV = interp1(dados.tempo, dados.MV, tempoUniforme, 'linear');

if all(~isnan(dados.SP))
    SP = interp1( ...
        dados.tempo, ...
        dados.SP, ...
        tempoUniforme, ...
        'previous');
else
    SP = NaN(size(tempoUniforme));
end

% Preencher valores faltantes após a interpolação
PV = fillmissing( ...
    PV, ...
    'linear', ...
    'EndValues', 'nearest');
MV = fillmissing( ...
    MV, ...
    'linear', ...
    'EndValues', 'nearest');

if all(~isnan(SP))
    SP = fillmissing( ...
        SP, ...
        'previous', ...
        'EndValues', 'nearest');
end

%% 1.4 Remover o valor médio dos sinais de MV e PV

PV0 = PV - mean(PV, 'omitnan');
MV0 = MV - mean(MV, 'omitnan');

if std(PV0, 'omitnan') <= eps
    error('O sinal PV apresenta variacao insuficiente.');
end

if std(MV0, 'omitnan') <= eps
    error('O sinal MV apresenta variacao insuficiente.');
end


%% ETAPA 2 - CALCULAR A FUNÇÃO DE CORRELAÇÃO CRUZADA

%% 2.1 Determinar o período dominante Tp
[Tp, frequenciaDominante, sinalPeriodo, qualidadeEspectral] = estimaPeriodoDominante(PV0, MV0, Ts);

%% 2.2 Calcular a tolerância Delta_phi (Critério Delta_phi = Tp/12)
DeltaPhi = Tp / 12;

% Converter a tolerância para número de amostras
DeltaPhiAmostras = max(1, ceil(DeltaPhi / Ts));

if qualidadeEspectral < 5
    warning(['O pico espectral utilizado para calcular Tp apresenta ', ...
        'baixa definição. Verifique se os sinais possuem uma ', ...
        'oscilacao periódica dominante.']);
end

%% 2.2 Calcular a CCF entre a MV e a PV para diferentes defasagens
% Equacao CCF: r_MV,PV(tau) = soma MV(k + tau)*PV(k)

% Definir as defasagens avaliadas
N = numel(MV0);
lags = -(N - 1):(N - 1);
ccf = zeros(size(lags));

for i = 1:numel(lags)
    tau = lags(i);
    if tau >= 0
        mvTrecho = MV0(1 + tau:N);
        pvTrecho = PV0(1:N - tau);
    else
        atraso = abs(tau);
        mvTrecho = MV0(1:N - atraso);
        pvTrecho = PV0(1 + atraso:N);
    end
    ccf(i) = sum(mvTrecho.*pvTrecho, 'omitnan');
end

% Normalizar a CCF
normalizador = N*std(MV0, 'omitnan')*std(PV0, 'omitnan');

if normalizador <= eps
    error('MV ou PV apresenta variacao insuficiente para calcular a CCF.');
end

ccf = ccf/normalizador;

% Converter as defasagens de amostras para segundos
lagsSegundos = lags * Ts;

% Localizar tau=0
indiceZero = find(lags == 0, 1);
r0 = ccf(indiceZero);

% Determinar o sinal da CCF na origem
sinalReferencia = sign(r0);

if sinalReferencia == 0
    inicioVizinhanca = max(1, indiceZero - 2);
    fimVizinhanca = min(numel(ccf), indiceZero + 2);
    vizinhanca = ccf(inicioVizinhanca:fimVizinhanca);
    vizinhanca = vizinhanca(vizinhanca ~= 0);
    if isempty(vizinhanca)
        error('Nao foi possivel definir o sinal da CCF em torno de tau = 0.');
    end
    sinalReferencia = sign(vizinhanca(1));
end

% Verificar se existe faixa suficiente alem de Delta_phi
maiorDefasagem = max(abs(lagsSegundos));

if DeltaPhi >= maiorDefasagem
    error(['Delta_phi e maior que a faixa de defasagens disponível. ', ...
        'Selecione um intervalo maior de dados.']);
end

%% 2.4 Determinar tau_l
% Primeiro cruzamento negativo por zero
zeroEsquerda = localizaCruzamentoNegativo(lagsSegundos, ccf, DeltaPhi);
tauL = abs(zeroEsquerda);

%% 2.5 Determinar tau_r
% Primeiro cruzamento positivo por zero
tauR = localizaCruzamentoPositivo(lagsSegundos, ccf, DeltaPhi);

%% 2.6 Determinar r_o
intervaloCentral = lagsSegundos >= zeroEsquerda & lagsSegundos <= tauR;

if ~any(intervaloCentral)
    error('O intervalo central da CCF não foi identificado.');
end

sinalR0 = sign(r0);

if sinalR0 == 0
    sinalR0 = sinalReferencia;
end

%% 2.7 Determinar r_opt
rOpt = sinalR0*max(abs(ccf(intervaloCentral)));

%% 2.8 Calcular Delta_tau
denominadorTau = tauL + tauR;

if denominadorTau <= eps
    error('Não foi possível calcular Delta_tau.');
end

DeltaTau = abs(tauL - tauR)/denominadorTau;

%% 2.9 Calcular Delta_rho
denominadorRho = abs(r0 + rOpt);

if denominadorRho <= eps
    DeltaRho = Inf;
else
    DeltaRho = abs(r0 - rOpt)/denominadorRho;
end

DeltaTau = min(max(DeltaTau, 0), 1);

if isfinite(DeltaRho)
    DeltaRho = min(max(DeltaRho, 0), 1);
end

%% 2.10 Classificar a CCF como par, ímpar ou indefinida
limitePar = 1/3;
limiteImpar = 2/3;

if DeltaTau <= limitePar && DeltaRho <= limitePar
    classificacao = 'CCF par';
    diagnostico = 'Grande probabilidade de ausência de agarramento na válvula de controle.';
elseif DeltaTau >= limiteImpar && DeltaRho >= limiteImpar
    classificacao = 'CCF ímpar';
    diagnostico = 'Possível ocorrência de agarramento na válvula de controle';
else
    classificacao = 'CCF na zona morta - paridade indefinida.';
    diagnostico = 'Não é possível definir se há presença ou não de agarramento na válvula de controle.';
end

%% ETAPA 3 - APRESENTAR O RESULTADO

%% 3.1 Apresentar o gráfico dos sinais da amostra avaliada
figSinais = figure('Name', 'Gráfico de Sinais da Amostra', 'Color', 'w');

plot(tempoUniforme, MV, 'Color', [31 79 216]/255, 'LineStyle', '-', 'LineWidth', 1.0, ...
    'DisplayName', 'MV');
hold on;
plot(tempoUniforme, PV, 'Color', [231 76 60]/255, 'LineStyle', '-', 'LineWidth', 1.0, ...
    'DisplayName', 'PV');
if all(~isnan(SP))
    plot(tempoUniforme, SP, 'Color', [46 204 113]/255, 'LineStyle', '--', 'LineWidth', 1.0, ...
        'DisplayName', 'SP');
end

hold off;
grid on;
box on;
xlabel('Tempo (s)');
ylabel('Amplitude');
title('Gráfico de Sinais da Amostra');
legend('Location', 'best');

%% 3.2 Apresentar os valores calculados para Tp, Delta_phi, tau_l, tau_r, r0, r_opt, Delta_tau e Delta_rho
fprintf('\n');
fprintf('============================================================\n');
fprintf('RESULTADO DA ANÁLISE POR CORRELACAO CRUZADA\n');
fprintf('============================================================\n');
fprintf('Arquivo: %s\n', arquivo);
fprintf('Numero de amostras analisadas: %d\n', N);
fprintf('Periodo de amostragem Ts: %.6f s\n', Ts);
fprintf('\n');
fprintf('PARAMETROS DA CCF\n');
fprintf('------------------------------------------------------------\n');
%fprintf('Sinal utilizado para calcular Tp: %s\n', sinalPeriodo);
%fprintf('Frequencia dominante fp: %.8f Hz\n', frequenciaDominante);
fprintf('Periodo dominante Tp: %.4f s\n', Tp);
fprintf('Delta_phi (Critério Tp/12): %.4f s\n', DeltaPhi);
%fprintf('Delta_phi em amostras: %d\n', DeltaPhiAmostras);
%fprintf('Qualidade do pico espectral: %.4f\n', qualidadeEspectral);
fprintf('tau_l: %.4f s\n', tauL);
fprintf('tau_r: %.4f s\n', tauR);
fprintf('r_0: %.6f\n', r0);
fprintf('r_opt: %.6f\n', rOpt);
fprintf('Delta_tau: %.6f\n', DeltaTau);
fprintf('Delta_rho: %.6f\n', DeltaRho);


%% 3.3 Apresentar o gráfico da Função de Correlação Cruzada
figCCF = figure('Name', 'Funcao de Correlacao Cruzada', 'Color', 'w');

plot(lagsSegundos, ccf, 'Color', [31 79 216]/255, 'LineWidth', 1.2, 'DisplayName', 'CCF');
hold on;
yline(0, '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
xline(0, '--', '\tau = 0', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
xline(-DeltaPhi, '--', '-\Delta_\phi', 'Color', [46 204 113]/255, 'FontSize', 14,'LineWidth', 1.2, ...
    'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(DeltaPhi, '--', '\Delta_\phi', 'Color', [46 204 113]/255, 'FontSize', 14,'LineWidth', 1.2, ...
    'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(-tauL, ':', '-\tau_l', 'Color', [31 79 216]/255, 'FontSize', 14,'LineWidth', 1.3, 'HandleVisibility', 'off');
xline(tauR, ':', '\tau_r', 'Color', [231 76 60]/255, 'FontSize', 14,'LineWidth', 1.3, 'HandleVisibility', 'off');
plot(0, r0, 'o', 'MarkerSize', 7, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', ...
    [0.3 0.3 0.3], 'DisplayName', 'r_0');
text(0, r0, 'r_0', 'Interpreter', 'tex', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
    'FontSize', 14, 'Color', [0.3 0.3 0.3]);
plot(-tauL, 0, 'o', 'MarkerSize', 7, 'MarkerEdgeColor', [31 79 216]/255, 'MarkerFaceColor', ...
    [31 79 216]/255, 'DisplayName', '-\tau_l');
text(-tauL, 0, '-\tau_l', 'Interpreter', 'tex', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
    'FontSize', 14, 'Color', [31 79 216]/255);
plot(tauR, 0, 'o', 'MarkerSize', 7, 'MarkerEdgeColor', [231 76 60]/255, 'MarkerFaceColor', ...
    [231 76 60]/255, 'DisplayName', '-\tau_r');
text(tauR, 0, '\tau_r', 'Interpreter', 'tex', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'FontSize', 14, 'Color', [231 76 60]/255);
hold off;
grid on;
box on;
xlabel('Defasagem \tau (s)');
ylabel('CCF normalizada');
title(sprintf(['%s | T_p = %.2f s | \\Delta_\\phi = %.2f s | ', ...
     '\\Delta_\\tau = %.3f | \\Delta_\\rho = %.3f'],classificacao, Tp, DeltaPhi, ...
    DeltaTau, DeltaRho));
legend('Location', 'best');
limiteGrafico = 1.5 * max([tauL, tauR, DeltaPhi]);
xlim([-limiteGrafico, limiteGrafico]);


%% 3.4 Apresentar o gráfico da Função de Correlação Cruzada
fprintf('\n');
fprintf('RESULTADO\n');
fprintf('------------------------------------------------------------\n');
fprintf('Classificação (par, ímpar ou indefinida): %s\n', classificacao);
fprintf('Diagnóstico: %s\n', diagnostico);
fprintf('============================================================\n');
fprintf('\n');

%% 3.5 Gerar relatório em PDF sem MATLAB Report Generator

% Nome do relatório
dataHoraAtual = datetime('now', 'Format','yyyy-MM-dd_HH-mm-ss');
dataAnalise = datetime('now', 'Format', 'dd.MM.yyyy HH:mm:ss');
nomePDF = fullfile(pwd, "Relatório_Técnico_CCF_" + arquivo + "_" + string(dataHoraAtual) + ".pdf");

% Apagar relatório com mesmo nome, caso exista
if isfile(nomePDF)
    delete(nomePDF);
end

% Criar Relatório

% Edição do conteúdo do relatório

margemfolha = sprintf([' \n', ' \n']);
titulo = sprintf('RELATÓRIO TÉCNICO DE ANÁLISE DE MALHA DE CONTROLE');
amostra = sprintf('%s', arquivo);
data = string(dataAnalise);
responsavel = sprintf('Márlon A. B. Damasceno');
instituicao = sprintf('UFOP - Universidade Federal de Ouro Preto');
cabecalho = sprintf(['Amostra: %s\n', 'Data: %s\n', 'Engenheiro Responsável: %s\n', ...
    '%s'], amostra, data, responsavel, instituicao);

textoParametros = sprintf([...
    'PARÂMETROS CALCULADOS PARA CCF                                                      Valor\n', ...
    '-------------------------------------------------------------------------------------------\n', ...
    'Período dominante (Tp)............................................................%.2f s\n', ...
    'Tolerância (Critério Delta_phi = Tp/12)........................................... %.2f s\n', ...
    'Primeiro cruzamento negativo, tau_l............................................... %.2f s\n', ...
    'Primeiro cruzamento positivo, tau_r...............................................%.2f s\n', ...
    'CCF na defasagem zero (r_0).......................................................%.2f\n', ...
    'Maior módulo da CCF, (r_opt)......................................................%.2f\n', ...
    'Índice Delta_tau..................................................................%.2f\n', ...
    'Índice Delta_rho..................................................................%.2f'], ...
    Tp, DeltaPhi, tauL, tauR, r0, rOpt, DeltaTau, DeltaRho);

% Configuração da página do relatório

paginaIdentificacao = figure('Name', 'Relatório - Identificação', 'Color', 'w', ...
    'Units', 'centimeters', 'Position', [2 2 21 29.7], 'PaperUnits', 'centimeters', ...
    'PaperSize', [21 29.7], 'PaperPosition', [0 0 21 29.7], 'Visible', 'off');

% Margem Superior
annotation(paginaIdentificacao, 'textbox', [0.08 0.88 0.84 0.10], 'String', ...
    margemfolha, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'white');

% Título
annotation(paginaIdentificacao, 'textbox', [0.08 0.88 0.84 0.12], 'String', ...
    titulo, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none');

% Cabeçalho
annotation(paginaIdentificacao, 'textbox', [0.10 0.86 0.84 0.04], 'String', ...
    cabecalho, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', ...
    10, 'FontWeight', 'normal', 'EdgeColor', 'none');

annotation(paginaIdentificacao, 'line', [0.08 0.92], [0.84 0.84], 'LineWidth', 1);

% Gráfico dos sinais da amostra
eixoOriginalSinais = findobj(figSinais, 'Type', 'axes');
eixoRelatorioSinais = copyobj(eixoOriginalSinais, paginaIdentificacao);

%set(eixoRelatorioSinais, 'Units', 'normalized', 'Position', [0.09 0.17 0.84 0.68], 'FontSize', 10);
set(eixoRelatorioSinais, 'Units', 'normalized', 'Position', [0.15 0.63 0.72 0.18], 'FontSize', 08);

% Parâmetros calculados dos sinais da amostra
annotation(paginaIdentificacao, 'textbox', [0.11 0.49 0.80 0.11], 'String', textoParametros, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontName', 'Courier New', ...
    'FontSize', 08, 'Interpreter', 'none', 'EdgeColor', 'white', 'LineWidth', 0.5, ...
    'BackgroundColor', 'white', 'Margin', 12);

% Título da seção do Gráfico da CCF
annotation(paginaIdentificacao, 'textbox', [0.05 0.44 0.90 0.02], 'String', 'GRÁFICO DA FUNÇÃO DE CORRELAÇÃO CRUZADA RESULTANTE ENTRE PV E MV', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, ...
    'FontWeight', 'bold', 'EdgeColor', 'white');

% Gráfico da CCF
eixoOriginalCCF = findobj(figCCF, 'Type', 'axes');
eixoRelatorioCCF = copyobj(eixoOriginalCCF, paginaIdentificacao);

set(eixoRelatorioCCF, 'Units', 'normalized', 'Position', [0.14 0.23 0.72 0.18], ...
    'FontSize', 08);

% Conclusão - Diagnóstico de agarramento via CCF
annotation(paginaIdentificacao, 'textbox',  [0.08 0.16 0.90 0.02], 'String', sprintf('RESULTADO DA ANÁLISE VIA CCF'), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, ...
    'FontWeight', 'bold', 'EdgeColor', 'white');

 annotation(paginaIdentificacao, 'textbox', [0.00 0.10 1.00 0.06], 'String', sprintf([ ...
     'CLASSIFICAÇÃO: %s\n\n', 'DIAGNÓSTICO: %s'], classificacao, diagnostico), 'HorizontalAlignment', 'left', ...
     'VerticalAlignment', 'middle', 'FontSize', 08, 'FontWeight', 'normal', 'Interpreter', 'none', 'EdgeColor', 'none', 'LineWidth', 1, ...
    'BackgroundColor', [255 255 0]/255, 'Margin', 12);

% Margem Inferior
annotation(paginaIdentificacao, 'textbox', [0.08 0.01 0.90 0.02], 'String', ...
    margemfolha, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'white');

exportgraphics(paginaIdentificacao, nomePDF, 'ContentType', 'vector');

close(paginaIdentificacao);

% Informa o local do arquivo gerado
fprintf('\n');
fprintf('============================================================\n');
fprintf('RELATÓRIO PDF GERADO COM SUCESSO\n');
fprintf('Arquivo: %s\n', nomePDF);
fprintf('============================================================\n\n');

% Abrir o PDF automaticamente
if ispc
    winopen(nomePDF);
elseif ismac
    system(sprintf('open "%s"', nomePDF));
else
    system(sprintf('xdg-open "%s" &', nomePDF));
end

%% FUNÇÕES UITILIZADAS NA ROTINA

% Função para cáuculo do período dominante Tp
function [Tp, frequenciaDominante, sinalSelecionado, qualidade] = estimaPeriodoDominante(PV, MV, Ts)
    PV = PV(:);
    MV = MV(:);
    if numel(PV) ~= numel(MV)
        error('Os sinais PV e MV devem possuir o mesmo tamanho.');
    end
    if numel(PV) < 10
        error('Existem poucas amostras para estimar o periodo dominante.');
    end
    if ~isfinite(Ts) || Ts <= 0
        error('O período de amostragem Ts e inválido.');
    end
    PV = fillmissing(PV, 'linear', 'EndValues', 'nearest');
    MV = fillmissing(MV, 'linear', 'EndValues', 'nearest');
    PV = PV - mean(PV, 'omitnan');
    MV = MV - mean(MV, 'omitnan');
    desvioPV = std(PV, 'omitnan');
    desvioMV = std(MV, 'omitnan');
    if desvioPV <= eps && desvioMV <= eps
        error('Não e possível calcular Tp porque PV e MV não apresentam variação suficiente.');
    end
    if desvioPV > eps
        PVnormalizada = PV / desvioPV;
    else
        PVnormalizada = zeros(size(PV));
    end
    if desvioMV > eps
        MVnormalizada = MV / desvioMV;
    else
        MVnormalizada = zeros(size(MV));
    end
    [frequenciaPV, qualidadePV] = calculaPicoEspectral(PVnormalizada, Ts);
    [frequenciaMV, qualidadeMV] = calculaPicoEspectral(MVnormalizada, Ts);
    if qualidadePV >= qualidadeMV
        frequenciaDominante = frequenciaPV;
        qualidade = qualidadePV;
        sinalSelecionado = 'PV';
    else
        frequenciaDominante = frequenciaMV;
        qualidade = qualidadeMV;
        sinalSelecionado = 'MV';
    end
    if ~isfinite(frequenciaDominante) || frequenciaDominante <= 0
        error('Não foi possível identificar uma frequência dominante válida.');
    end
    Tp = 1 / frequenciaDominante;
end

% Função para cálculo da frequência dominante fp
function [frequenciaDominante, qualidade] = calculaPicoEspectral(sinal, Ts)
    sinal = sinal(:);
    N = numel(sinal);
    if std(sinal, 'omitnan') <= eps
        frequenciaDominante = NaN;
        qualidade = 0;
        return;
    end
    Fs = 1 / Ts;
    duracao = (N - 1) * Ts;
    if duracao <= 0
        error('A duracao do sinal e inválida.');
    end
    n = (0:N - 1)';
    janela = 0.5 - 0.5*cos(2*pi*n/(N - 1));
    sinalComJanela = sinal .* janela;
    NFFT = 2^nextpow2(4*N);
    espectro = fft(sinalComJanela, NFFT);
    potencia = abs(espectro(1:floor(NFFT/2) + 1)).^2;
    frequencias = (0:floor(NFFT/2))' * Fs/NFFT;
    frequenciaMinima = 2 / duracao;
    frequenciaMaxima = Fs / 4;
    faixaValida = frequencias >= frequenciaMinima & frequencias <= frequenciaMaxima;
    if ~any(faixaValida)
        error('O intervalo analisado e insuficiente para identificar uma oscilacão periódica valida.');
    end
    indicesValidos = find(faixaValida);
    potenciaValida = potencia(indicesValidos);
    [pico, indiceLocal] = max(potenciaValida);
    indicePico = indicesValidos(indiceLocal);
    deslocamento = 0;

    if indicePico > 1 && indicePico < numel(potencia)
        p1 = potencia(indicePico - 1);
        p2 = potencia(indicePico);
        p3 = potencia(indicePico + 1);
        denominador = p1 - 2*p2 + p3;
        if abs(denominador) > eps
            deslocamento = 0.5*(p1 - p3)/denominador;
        end
    end
    resolucaoFrequencia = Fs/NFFT;
    frequenciaDominante = ...
        frequencias(indicePico) + deslocamento*resolucaoFrequencia;
    nivelEspectral = median(potenciaValida);
    if nivelEspectral <= eps
        qualidade = Inf;
    else
        qualidade = pico/nivelEspectral;
    end
end

% Função para calcular o pico de espectro da frequência dominante (pré-FFT)
function [frequencias, potencia] = calculaEspectro(sinal, Ts)
    sinal = sinal(:);
    sinal = fillmissing(sinal, 'linear', 'EndValues', 'nearest');
    sinal = sinal - mean(sinal, 'omitnan');
    N = numel(sinal);
    Fs = 1/Ts;
    n = (0:N - 1)';
    janela = 0.5 - 0.5*cos(2*pi*n/(N - 1));
    NFFT = 2^nextpow2(4*N);
    espectro = fft(sinal .* janela, NFFT);
    potencia = abs(espectro(1:floor(NFFT/2) + 1)).^2;
    frequencias = (0:floor(NFFT/2))' * Fs/NFFT;
end

% Função para calcular tau_r
function tauR = localizaCruzamentoPositivo(lagsSegundos, ccf, DeltaPhi)
    indiceInicial = find(lagsSegundos >= DeltaPhi, 1, 'first');

    if isempty(indiceInicial) || indiceInicial >= numel(ccf)
        error('NÃo existe faixa positiva suficiente fora da tolerância Delta_phi.');
    end

    while indiceInicial < numel(ccf) && ccf(indiceInicial) == 0
        indiceInicial = indiceInicial + 1;
    end

    if indiceInicial >= numel(ccf)
        error('Não foi possível determinar o sinal da CCF fora de Delta_phi no lado positivo.');
    end
    sinalInicial = sign(ccf(indiceInicial));
    indiceCruzamento = [];
    for i = indiceInicial + 1:numel(ccf)
        sinalAtual = sign(ccf(i));
        if ccf(i) == 0 || (sinalAtual ~= 0 && sinalAtual ~= sinalInicial)
            indiceCruzamento = i;
            break;
        end
    end

    if isempty(indiceCruzamento)
        error('Não foi encontrado cruzamento por zero para defasagens positivas fora de Delta_phi.');
    end
    tauR = interpolaCruzamentoZero(lagsSegundos(indiceCruzamento - 1), ccf(indiceCruzamento - 1), ...
        lagsSegundos(indiceCruzamento), ccf(indiceCruzamento));
    if tauR < DeltaPhi
        error('O cruzamento positivo identificado está dentro da zona de tolerância Delta_phi.');
    end
end

% Função para calcular tau_l
function zeroEsquerda = localizaCruzamentoNegativo(lagsSegundos, ccf, DeltaPhi)

    indiceInicial = find(lagsSegundos <= -DeltaPhi, 1, 'last');

    if isempty(indiceInicial) || indiceInicial <= 1
        error('Não existe faixa negativa suficiente fora da tolerância Delta_phi.');
    end

    while indiceInicial > 1 && ccf(indiceInicial) == 0
        indiceInicial = indiceInicial - 1;
    end

    if indiceInicial <= 1
        error('Não foi possível determinar o sinal da CCF fora de Delta_phi no lado negativo.');
    end

    sinalInicial = sign(ccf(indiceInicial));
    indiceCruzamento = [];

    for i = indiceInicial - 1:-1:1
        sinalAtual = sign(ccf(i));
        if ccf(i) == 0 || (sinalAtual ~= 0 && sinalAtual ~= sinalInicial)
            indiceCruzamento = i;
            break;
        end
    end

    if isempty(indiceCruzamento)
        error('Não foi encontrado cruzamento por zero para defasagens negativas fora de Delta_phi.');
    end

    zeroEsquerda = interpolaCruzamentoZero(lagsSegundos(indiceCruzamento), ...
        ccf(indiceCruzamento), lagsSegundos(indiceCruzamento + 1), ccf(indiceCruzamento + 1));

    if zeroEsquerda > -DeltaPhi
        error('O cruzamento negativo identificado está dentro da zona de tolerancia Delta_phi.');
    end
end


function xZero = interpolaCruzamentoZero(x1, y1, x2, y2)
    if y1 == 0
        xZero = x1;
    elseif y2 == 0
        xZero = x2;
    elseif y2 == y1
        xZero = (x1 + x2)/2;
    else
        xZero = x1 - y1*(x2 - x1)/(y2 - y1);
    end
end
