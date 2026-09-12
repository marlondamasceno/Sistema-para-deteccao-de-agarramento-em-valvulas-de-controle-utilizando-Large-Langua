%% ============================================================
% CÁLCULO DA FFT, FREQUÊNCIA DOMINANTE E VALOR RMS
%
% Entrada:
%   Arquivo CSV contendo as colunas:
%   1 - tempo
%   2 - SP
%   3 - PV
%   4 - MV
%
% Saída:
%   1 - Gráfico do espectro de frequência
%   2 - Frequência dominante
%   3 - Valor RMS da série temporal
%   4 - Arquivo CSV contendo os resultados
%% ============================================================

clear;
clc;
close all;

%% 1. CONFIGURAÇÃO

arquivoEntrada = 'dados.csv';

% Sinal que será analisado:
% 'SP', 'PV' ou 'MV'
sinalSelecionado = 'PV';

%% 2. VERIFICAÇÃO DO ARQUIVO

if exist(arquivoEntrada, 'file') ~= 2
    error('O arquivo "%s" não foi encontrado.', arquivoEntrada);
end

%% 3. LEITURA DO ARQUIVO CSV

dados = readtable(arquivoEntrada);

if width(dados) < 4
    error(['O arquivo deve possuir pelo menos quatro colunas: ', ...
           'tempo, SP, PV e MV.']);
end

%% 4. EXTRAÇÃO DAS COLUNAS

tempo = dados{:,1};
SP    = dados{:,2};
PV    = dados{:,3};
MV    = dados{:,4};

%% 5. SELEÇÃO DA SÉRIE TEMPORAL

switch upper(sinalSelecionado)

    case 'SP'
        x = SP;

    case 'PV'
        x = PV;

    case 'MV'
        x = MV;

    otherwise
        error('Selecione um sinal válido: SP, PV ou MV.');

end

%% 6. CONVERSÃO DOS VETORES PARA COLUNAS

tempo = tempo(:);
x = x(:);

%% 7. REMOÇÃO DE AMOSTRAS INVÁLIDAS

indicesValidos = isfinite(tempo) & isfinite(x);

tempo = tempo(indicesValidos);
x = x(indicesValidos);

%% 8. VERIFICAÇÃO DA QUANTIDADE DE AMOSTRAS

N = length(x);

if N < 2
    error('A série temporal deve possuir pelo menos duas amostras válidas.');
end

%% 9. CÁLCULO DO PERÍODO DE AMOSTRAGEM

diferencasTempo = diff(tempo);

if any(diferencasTempo <= 0)
    error('O vetor de tempo deve ser estritamente crescente.');
end

Ts = mean(diferencasTempo);

%% 10. CÁLCULO DA FREQUÊNCIA DE AMOSTRAGEM

fs = 1/Ts;

%% 11. CÁLCULO DO VALOR RMS DA SÉRIE TEMPORAL

valorRMS = sqrt(mean(x.^2));

%% 12. CÁLCULO DA FFT NORMALIZADA

X = fft(x)/N;

%% 13. CONSTRUÇÃO DO ESPECTRO UNILATERAL

N_half = floor(N/2) + 1;

X_half = X(1:N_half);

f = (0:N_half-1)'*(fs/N);

magnitude = abs(X_half);

%% 14. CORREÇÃO DA MAGNITUDE DO ESPECTRO UNILATERAL

if rem(N,2) == 0

    % Número par de amostras:
    % não dobrar as componentes DC e Nyquist
    if length(magnitude) > 2
        magnitude(2:end-1) = 2*magnitude(2:end-1);
    end

else

    % Número ímpar de amostras:
    % não dobrar apenas a componente DC
    if length(magnitude) > 1
        magnitude(2:end) = 2*magnitude(2:end);
    end

end

%% 15. DETERMINAÇÃO DA FREQUÊNCIA DOMINANTE

% A primeira posição, correspondente a 0 Hz, é desconsiderada
[amplitudeDominante, indiceRelativo] = max(magnitude(2:end));

% Correção do índice devido à retirada da primeira posição
indiceDominante = indiceRelativo + 1;

freqDominante = f(indiceDominante);

%% 16. CRIAÇÃO DA TABELA DE RESULTADOS

nomeSinal = {upper(sinalSelecionado)};

resultados = table( ...
    nomeSinal, ...
    N, ...
    Ts, ...
    fs, ...
    freqDominante, ...
    amplitudeDominante, ...
    valorRMS, ...
    'VariableNames', { ...
        'Sinal', ...
        'Numero_Amostras', ...
        'Periodo_Amostragem_s', ...
        'Frequencia_Amostragem_Hz', ...
        'Frequencia_Dominante_Hz', ...
        'Amplitude_Dominante', ...
        'Valor_RMS' ...
    } ...
);

%% 17. DEFINIÇÃO DO ARQUIVO DE SAÍDA

[pastaEntrada, nomeArquivo, ~] = fileparts(arquivoEntrada);

if isempty(pastaEntrada)
    pastaEntrada = pwd;
end

nomeArquivoSaida = [nomeArquivo, '_resultados_fft_rms.csv'];

arquivoSaida = fullfile( ...
    pastaEntrada, ...
    nomeArquivoSaida ...
);

%% 18. SALVAMENTO DOS RESULTADOS EM CSV

writetable( ...
    resultados, ...
    arquivoSaida, ...
    'Delimiter', ';' ...
);

%% 19. APRESENTAÇÃO DO ESPECTRO DE FREQUÊNCIA

figure('Color', 'w');

plot( ...
    f, ...
    magnitude, ...
    'b-', ...
    'LineWidth', 1.5 ...
);

hold on;

plot( ...
    freqDominante, ...
    amplitudeDominante, ...
    'ro', ...
    'MarkerSize', 7, ...
    'MarkerFaceColor', 'r' ...
);

grid on;
box on;

xlabel('Frequência (Hz)');
ylabel('Magnitude');

title([ ...
    'Espectro de Frequência -- FFT do sinal ', ...
    upper(sinalSelecionado) ...
]);

legend( ...
    'Espectro unilateral', ...
    'Frequência dominante', ...
    'Location', ...
    'best' ...
);

%% 20. APRESENTAÇÃO DOS RESULTADOS NO MATLAB

fprintf('\n');
fprintf('=============================================\n');
fprintf('RESULTADOS DA ANÁLISE\n');
fprintf('=============================================\n');
fprintf('Sinal analisado: %s\n', upper(sinalSelecionado));
fprintf('Número de amostras: %d\n', N);
fprintf('Período médio de amostragem: %.6f s\n', Ts);
fprintf('Frequência de amostragem: %.6f Hz\n', fs);
fprintf('Frequência dominante: %.6f Hz\n', freqDominante);
fprintf('Amplitude dominante: %.6f\n', amplitudeDominante);
fprintf('Valor RMS da série temporal: %.6f\n', valorRMS);
fprintf('Arquivo de resultados:\n%s\n', arquivoSaida);
fprintf('=============================================\n');