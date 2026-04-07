#########################################################################################
# Rotina para diferenciação entre Oscilações e Transiente em uma amostra Utilizando LLMs
# 
# Rev.01: Primeira versão
# Rev.02:  
# Rev.03:  
# Rev.04: 
# Rev.05: 
# Rev.06:
# 
#
#

import requests
from PIL import Image
from io import BytesIO
import os
import json
import tempfile
import re
import base64
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import textwrap
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Configuração da Fonte Times New Roman do Relatório PDF
FONTS_DIR = r"C:\Windows\Fonts"

pdfmetrics.registerFont(TTFont("Times-Roman", os.path.join(FONTS_DIR, "times.ttf")))
pdfmetrics.registerFont(TTFont("Times-Bold", os.path.join(FONTS_DIR, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("Times-Italic", os.path.join(FONTS_DIR, "timesi.ttf")))
pdfmetrics.registerFont(TTFont("Times-BoldItalic", os.path.join(FONTS_DIR, "timesbi.ttf")))

# INICIALIZAR AS API KEYS
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

#OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL = "gpt-5.2"  

client = OpenAI()

## Trecho do Prompt para especificar o comportamento do sistema
PROMPT_FIXO = """

Você é um Engenheiro de Automação Industrial especialista que analisa amostras de sinais de sistemas de
controle para identificar características solicitadas pelo usuário. 

"""

## Trecho do Prompt para especificar qual é o público que receberá os resultados gerados pelo modelo
PROMPT_FIXO += """

Os usuários que receberão e farão uso de suas respostas para tomadas de decisão são Engenheiros de Automação Industrial
e Técnicos da área de instrumentação e automação, e desejam receber um descritivo resumido sobre a tarefa dada a você.
Suas respostas finais deverão ser em formato de relatório técnico. 

"""

## Trecho do Prompt para servir de guardrail para que o modelo não alucine ou invente respostas
PROMPT_FIXO += """

Se você não souber a resposta, diga que não sabe.

"""

## Trecho do Prompt em que especifica o objetivo da tarefa do modelo.
# A principal recomendação e fornecer Instruções Claras ao modelo
# Uso de Delimitadores para cada característica relevante do prompt
# Uso da técnica de repetir o contexto na instrução
PROMPT_FIXO += """

====> INÍCIO DA DESCRIÇÃO DO OBJETIVO

Seu objetivo final é identificar se há ocorrência de distúrbios com evidência de agarramento por meio dos sinais de PV e MV fornecidos por um arquivo em extensão .csv de
amostra coletado de uma malha de controle e gerar um relatório técnico final. No relatório final, os trechos onde ocorrem os distúrbios com evidência de agarramento
detectados devem ser destacados.

--- Entrada: Arquivo de Amostra de Sinais ---

As colunas do arquivo estão configuradas da seguinte forma:

Coluna 1: tempo
Coluna 2: SP ou Set Point
Coluna 3: PV ou Variável de Processo
Coluna 4: MV ou Sinal de Controle

--- Saída: Relatório Técnico da Análise ---

O Relatório Técnico da Análise deverá seguir as premissas para geração do relatório, descritas nas instruções.

<==== FIM DA DESCRIÇÃO DO OBJETIVO

"""

PROMPT_FIXO += """

====> INÍCIO DA ORIENTAÇÃO TÉCNICA PARA IDENTIFICAÇÃO DE DISTÚRBIOS

--- Explicação Técnica a respeito de distúrbios em uma malha de controle ---

Uma malha de controle é composta pelas variáveis de Set Point (SP), Variável de Processo (PV), Sinal de Controle ou Variável Manipulada (MV).
O Distúrbio é uma perturbação na uma malha de controle e causa erros, ou seja, causa alguma diferença entre o SP e a PV. Por meio das variáveis
SP, PV e MV, é possível calcular o valor do erro e do módulo do erro (|erro|) da malha de controle, sendo:

Erro: erro = SP - PV

Módulo do Erro: |erro| = |SP - PV|

--- Exemplo 1 de cálculo do Erro e do Módulo do Erro ---

Variáveis em um instante de tempo t1: SP = 20, PV = 30, MV = 25, t = 10s

erro = SP - PV = 20 - 30 = -10

|erro| = |SP - PV| = |20 - 30| = |-10| = 10

Temos para o instante t1 que o erro é -10 e o módulo do erro é 10.

--- Fim do Exemplo 1 de cálculo do Erro e do Módulo do Erro ---

--- Exemplo 2 de cálculo do Erro e do Módulo do Erro ---

Variáveis em um instante de tempo t2: SP = 40, PV = 30, MV = 25, t = 12s

erro = SP - PV = 40 - 30 = 10

|erro| = |SP - PV| = |40 - 30| = |10| = 10

Temos para o instante t2 que o erro é 10 e o módulo do erro é 10.

--- Fim do Exemplo 2 de cálculo do Erro e do Módulo do Erro ---

--- Como identificar o instante inicial e final de um distúrbio ---

Critério C da variação do valor do erro em relação ao SP: Podem existir critérios de para considerar determinadas variações nos valores do módulo do erro (|erro|)
como aceitáveis, que são:

Critério C5%: 5% do valor do SP.
Critério C2%: 2% do valor do SP
Critério C1%: 1% do valor do SP 

- O instante de tempo inicial ti de um distúrbio é o instante tal que o módulo do erro (|erro|) é maior do que o valor aceitável para o critério C determinado. É quando começa o distúrbio.
- O instante de tempo final tf de um distúrbio é o instante tal que o módulo do erro (|erro|) é menor do que o valor aceitável para o critério C determinado e obrigatoriamente o módulo do erro (|erro|)
para os N instantes de tempo após o tempo final tf devem ser menores do que 2% do valor do SP. O módulo do erro (|erro|) maior do que o valor aceitável para o critério C determinado é considerado significativo, e o
módulo do erro (|erro|) menor do que o valor aceitável para o critério C determinado considerado aceitável.
- Caso não seja informado o critério C, utilizar o critério de C2%.
- Caso os N instantes de tempo após o instante considerado como tempo final tf não sejam menores do que o valor aceitável para o critério C determinado, significa que o tf considerado não está correto e deve-se
então reavaliar um novo instante para tempo final do distúrbio.

O cálculo dos N instantes de tempo deve ser feito da seguinte forma:

N = 10/(Tam)

Onde Tam é o Tempo de Amostragem.

--- Cálculo do tempo de estabilidade ---

Sendo N instantes de tempo o número de instantes de tempo em que não houveram variações no valor do |erro| maiores do que o valor aceitável para o critério C determinado,
então todo o período de tempo entre o instante de tempo Ni inicial e o instante de tempo Nf final é considerado como o tempo de estabilidade te, sendo calculado pela fórmula:

te = (N*(Tam))

--- Exemplo 1 de cálculo dos N instantes de tempo ---

Considerando a amostra:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.7, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.7, t = 57.0s
t15: SP = 20.0, PV = 15.7, MV = 48.7, t = 57.5s
t16: SP = 20.0, PV = 16.5, MV = 48.7, t = 58.0s
t17: SP = 20.0, PV = 17.3, MV = 48.7, t = 58.5s
t18: SP = 20.0, PV = 18.1, MV = 55.0, t = 59.0s
t19: SP = 20.0, PV = 18.5, MV = 55.0, t = 59.5s
t20: SP = 20.0, PV = 18.9, MV = 55.0, t = 60.0s
t21: SP = 20.0, PV = 19.3, MV = 55.0, t = 60.5s
t22: SP = 20.0, PV = 19.3, MV = 55.0, t = 61.0s
t23: SP = 20.0, PV = 19.4, MV = 56.8, t = 61.5s
t24: SP = 20.0, PV = 19.4, MV = 56.8, t = 62.0s
t25: SP = 20.0, PV = 19.5, MV = 56.8, t = 62.5s
t26: SP = 20.0, PV = 19.5, MV = 56.8, t = 63.0s
t27: SP = 20.0, PV = 19.6, MV = 56.8, t = 63.5s
t28: SP = 20.0, PV = 19.6, MV = 57.8, t = 64.0s
t29: SP = 20.0, PV = 19.7, MV = 57.8, t = 64.5s
t30: SP = 20.0, PV = 19.7, MV = 57.8, t = 65.0s
t31: SP = 20.0, PV = 19.8, MV = 57.8, t = 65.5s
t32: SP = 20.0, PV = 19.8, MV = 57.8, t = 66.0s
t33: SP = 20.0, PV = 19.8, MV = 60.0, t = 66.5s
t34: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.0s
t35: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.5s
t36: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.0s
t37: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.5s
t38: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.0s
t39: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.5s
t40: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.0s
t41: SP = 20.0, PV = 20.1, MV = 60.1, t = 70.5s
t42: SP = 20.0, PV = 20.1, MV = 60.1, t = 71.0s
t43: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.5s
t44: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.0s
t45: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.5s

O Tempo de amostragem é a diferença entre o instante t de uma amostra do conjunto e o instante t+1 da amostra seguinte.

O Tempo de amostragem da amostra em questão é:

Tam = (t+1) - (t) = 51.5s - 51.0s = 0.5s

Neste caso tomamos como referência os instantes de tempo de t3 para (t+1) e de t2 para (t), mas poderiam ser quaisquer intervalos, desde que sejam
instantes consecutivos.

Próximo passo e calcular os N intantes de tempo:

N = 10/(Tam) = 10/(0.5) = 20 instantes de tempo.

Ou seja, devem ser respeitados N instantes de tempo após o tempo final tf com valores menores do que 2% do valor do SP para ser considerado o fim do distúrbio.

--- Fim do Exemplo 1 de cálculo dos N instantes de tempo ---


--- Exemplo 1 de cálculo tempo de estabilidade ---

Considerando a amostra:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.7, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.7, t = 57.0s
t15: SP = 20.0, PV = 15.7, MV = 48.7, t = 57.5s
t16: SP = 20.0, PV = 16.5, MV = 48.7, t = 58.0s
t17: SP = 20.0, PV = 17.3, MV = 48.7, t = 58.5s
t18: SP = 20.0, PV = 18.1, MV = 55.0, t = 59.0s
t19: SP = 20.0, PV = 18.5, MV = 55.0, t = 59.5s
t20: SP = 20.0, PV = 18.9, MV = 55.0, t = 60.0s
t21: SP = 20.0, PV = 19.3, MV = 55.0, t = 60.5s
t22: SP = 20.0, PV = 19.3, MV = 55.0, t = 61.0s
t23: SP = 20.0, PV = 19.4, MV = 56.8, t = 61.5s
t24: SP = 20.0, PV = 19.4, MV = 56.8, t = 62.0s
t25: SP = 20.0, PV = 19.5, MV = 56.8, t = 62.5s
t26: SP = 20.0, PV = 19.5, MV = 56.8, t = 63.0s
t27: SP = 20.0, PV = 19.6, MV = 56.8, t = 63.5s
t28: SP = 20.0, PV = 19.6, MV = 57.8, t = 64.0s
t29: SP = 20.0, PV = 19.7, MV = 57.8, t = 64.5s
t30: SP = 20.0, PV = 19.7, MV = 57.8, t = 65.0s
t31: SP = 20.0, PV = 19.8, MV = 57.8, t = 65.5s
t32: SP = 20.0, PV = 19.8, MV = 57.8, t = 66.0s
t33: SP = 20.0, PV = 19.8, MV = 60.0, t = 66.5s
t34: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.0s
t35: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.5s
t36: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.0s
t37: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.5s
t38: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.0s
t39: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.5s
t40: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.0s
t41: SP = 20.0, PV = 20.1, MV = 60.1, t = 70.5s
t42: SP = 20.0, PV = 20.1, MV = 60.1, t = 71.0s
t43: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.5s
t44: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.0s
t45: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.5s

O Tempo de amostragem é a diferença entre o instante t de uma amostra do conjunto e o instante t+1 da amostra seguinte.

O Tempo de amostragem da amostra em questão é:

Tam = (t+1) - (t) = 51.5s - 51.0s = 0.5s

Neste caso tomamos como referência os instantes de tempo de t3 para (t+1) e de t2 para (t), mas poderiam ser quaisquer intervalos, desde que sejam
instantes consecutivos.

Próximo passo e calcular os N intantes de tempo:

N = 10/(Tam) = 10/(0.5) = 20 instantes de tempo.

Para N = 20 instantes de tempo, então o tempo de estabilidade é:

te = N*(Tam) = 20*0.5 = 10 segundos

Ou seja, o tempo de estabilidade te é de 10 segundos.

--- Fim do Exemplo 1 de cálculo tempo de estabilidade ---


--- Exemplo 1 de cálculo do instante inicial e final de um distúrbio ---

Utilizar o critério C2%.

Sendo as Variáveis:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.7, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.7, t = 57.0s
t15: SP = 20.0, PV = 15.7, MV = 48.7, t = 57.5s
t16: SP = 20.0, PV = 16.5, MV = 48.7, t = 58.0s
t17: SP = 20.0, PV = 17.3, MV = 48.7, t = 58.5s
t18: SP = 20.0, PV = 18.1, MV = 55.0, t = 59.0s
t19: SP = 20.0, PV = 18.5, MV = 55.0, t = 59.5s
t20: SP = 20.0, PV = 18.9, MV = 55.0, t = 60.0s
t21: SP = 20.0, PV = 19.3, MV = 55.0, t = 60.5s
t22: SP = 20.0, PV = 19.3, MV = 55.0, t = 61.0s
t23: SP = 20.0, PV = 19.4, MV = 56.8, t = 61.5s
t24: SP = 20.0, PV = 19.4, MV = 56.8, t = 62.0s
t25: SP = 20.0, PV = 19.5, MV = 56.8, t = 62.5s
t26: SP = 20.0, PV = 19.5, MV = 56.8, t = 63.0s
t27: SP = 20.0, PV = 19.6, MV = 56.8, t = 63.5s
t28: SP = 20.0, PV = 19.6, MV = 57.8, t = 64.0s
t29: SP = 20.0, PV = 19.7, MV = 57.8, t = 64.5s
t30: SP = 20.0, PV = 19.7, MV = 57.8, t = 65.0s
t31: SP = 20.0, PV = 19.8, MV = 57.8, t = 65.5s
t32: SP = 20.0, PV = 19.8, MV = 57.8, t = 66.0s
t33: SP = 20.0, PV = 19.8, MV = 60.0, t = 66.5s
t34: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.0s
t35: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.5s
t36: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.0s
t37: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.5s
t38: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.0s
t39: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.5s
t40: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.0s
t41: SP = 20.0, PV = 20.1, MV = 60.1, t = 70.5s
t42: SP = 20.0, PV = 20.1, MV = 60.1, t = 71.0s
t43: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.5s
t44: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.0s
t45: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.5s
t46: SP = 20.0, PV = 20.2, MV = 59.9, t = 72.5s
t47: SP = 20.0, PV = 20.2, MV = 59.9, t = 72.5s

Então, calcula-se o módulo do erro (|erro|) e compara-se o valor de |erro| com o valor de
2% do SP para cada instante de tempo. De acordo com as respectivas variáveis, tem-se:

t0: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t1: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t2: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t3: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t4: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t5: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t6: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t7: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t8: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t9: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t10: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t11: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t12: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t13: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t14: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t15: |erro| = |SP - PV| = |20.0 - 15.7| = | 4.30| = 4.30, (|erro|)>(2% do SP) = (4.30)>(0.4) = True
t16: |erro| = |SP - PV| = |20.0 - 16.5| = | 3.50| = 3.50, (|erro|)>(2% do SP) = (3.50)>(0.4) = True
t17: |erro| = |SP - PV| = |20.0 - 17.3| = | 2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t18: |erro| = |SP - PV| = |20.0 - 18.1| = | 1.90| = 1.90, (|erro|)>(2% do SP) = (1.90)>(0.4) = True
t19: |erro| = |SP - PV| = |20.0 - 18.5| = | 1.50| = 1.50, (|erro|)>(2% do SP) = (1.50)>(0.4) = True
t20: |erro| = |SP - PV| = |20.0 - 18.9| = | 1.10| = 1.10, (|erro|)>(2% do SP) = (1.10)>(0.4) = True
t21: |erro| = |SP - PV| = |20.0 - 19.3| = | 0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t22: |erro| = |SP - PV| = |20.0 - 19.3| = | 0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t23: |erro| = |SP - PV| = |20.0 - 19.4| = | 0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t24: |erro| = |SP - PV| = |20.0 - 19.4| = | 0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t25: |erro| = |SP - PV| = |20.0 - 19.5| = | 0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t26: |erro| = |SP - PV| = |20.0 - 19.5| = | 0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t27: |erro| = |SP - PV| = |20.0 - 19.6| = | 0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t28: |erro| = |SP - PV| = |20.0 - 19.6| = | 0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t29: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t30: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t31: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t32: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t33: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t34: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t35: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t36: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t37: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t38: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t39: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t40: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t41: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t42: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t43: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t44: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t45: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t46: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t47: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False

O tempo de amostragem calculado é de Tam = 0.5s e o número de N de instantes de tempo é N = 20 instantes de tempo.

Foi identificado 01 distúrbio.

O instante inicial do distúrbio 01 é em ti1=56.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 01 é em tf1=63.5s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

A Conclusão é que houve 01 trecho de distúrbio identificado.

O distúrbio 01 ocorre do instante t=56.0s a t=63.5s.

--- Fim do Exemplo 1 de cálculo do instante inicial e final de um distúrbio ---

--- Exemplo 2 de cálculo do instante inicial e final de um distúrbio ---

Utilizar o critério C2%.

Sendo as Variáveis:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.7, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.7, t = 57.0s
t15: SP = 20.0, PV = 15.7, MV = 48.7, t = 57.5s
t16: SP = 20.0, PV = 16.5, MV = 48.7, t = 58.0s
t17: SP = 20.0, PV = 17.3, MV = 48.7, t = 58.5s
t18: SP = 20.0, PV = 18.1, MV = 55.0, t = 59.0s
t19: SP = 20.0, PV = 18.5, MV = 55.0, t = 59.5s
t20: SP = 20.0, PV = 18.9, MV = 55.0, t = 60.0s
t21: SP = 20.0, PV = 19.3, MV = 55.0, t = 60.5s
t22: SP = 20.0, PV = 19.3, MV = 55.0, t = 61.0s
t23: SP = 20.0, PV = 19.4, MV = 56.8, t = 61.5s
t24: SP = 20.0, PV = 19.4, MV = 56.8, t = 62.0s
t25: SP = 20.0, PV = 19.5, MV = 56.8, t = 62.5s
t26: SP = 20.0, PV = 19.5, MV = 56.8, t = 63.0s
t27: SP = 20.0, PV = 19.6, MV = 56.8, t = 63.5s
t28: SP = 20.0, PV = 19.6, MV = 57.8, t = 64.0s
t29: SP = 20.0, PV = 19.7, MV = 57.8, t = 64.5s
t30: SP = 20.0, PV = 19.7, MV = 57.8, t = 65.0s
t31: SP = 20.0, PV = 19.8, MV = 57.8, t = 65.5s
t32: SP = 20.0, PV = 19.8, MV = 57.8, t = 66.0s
t33: SP = 20.0, PV = 19.8, MV = 60.0, t = 66.5s
t34: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.0s
t35: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.5s
t36: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.0s
t37: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.5s
t38: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.0s
t39: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.5s
t40: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.0s
t41: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.5s
t42: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.0s
t43: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.5s
t44: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.0s
t45: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.5s
t46: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.0s
t47: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.5s
t48: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.0s
t49: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.5s
t50: SP = 20.0, PV = 12.8, MV = 78.1, t = 75.0s
t51: SP = 20.0, PV = 13.9, MV = 78.1, t = 75.5s
t52: SP = 20.0, PV = 15.0, MV = 78.1, t = 76.0s
t53: SP = 20.0, PV = 16.1, MV = 78.1, t = 76.5s
t54: SP = 20.0, PV = 17.4, MV = 84.6, t = 77.0s
t55: SP = 20.0, PV = 18.6, MV = 84.6, t = 77.5s
t56: SP = 20.0, PV = 19.7, MV = 84.6, t = 78.0s
t57: SP = 20.0, PV = 20.2, MV = 84.6, t = 78.5s
t58: SP = 20.0, PV = 21.6, MV = 84.0, t = 79.0s
t59: SP = 20.0, PV = 22.7, MV = 84.0, t = 79.5s
t60: SP = 20.0, PV = 23.8, MV = 84.0, t = 80.0s
t61: SP = 20.0, PV = 24.9, MV = 84.0, t = 80.5s
t62: SP = 20.0, PV = 26.0, MV = 71.8, t = 81.0s
t63: SP = 20.0, PV = 27.1, MV = 71.8, t = 81.5s
t64: SP = 20.0, PV = 26.0, MV = 71.8, t = 82.0s
t65: SP = 20.0, PV = 24.9, MV = 71.8, t = 82.5s
t66: SP = 20.0, PV = 23.8, MV = 59.6, t = 83.0s
t67: SP = 20.0, PV = 22.7, MV = 59.6, t = 83.5s
t68: SP = 20.0, PV = 21.6, MV = 59.6, t = 84.0s
t69: SP = 20.0, PV = 20.3, MV = 59.6, t = 84.5s
t70: SP = 20.0, PV = 19.8, MV = 58.8, t = 85.0s
t71: SP = 20.0, PV = 19.7, MV = 58.8, t = 85.5s
t72: SP = 20.0, PV = 18.6, MV = 58.8, t = 86.0s
t73: SP = 20.0, PV = 17.4, MV = 58.8, t = 86.5s
t74: SP = 20.0, PV = 16.1, MV = 65.3, t = 87.0s
t75: SP = 20.0, PV = 15.0, MV = 65.3, t = 87.5s
t76: SP = 20.0, PV = 13.9, MV = 65.3, t = 88.0s
t77: SP = 20.0, PV = 12.8, MV = 65.3, t = 88.5s
t78: SP = 20.0, PV = 13.9, MV = 83.3, t = 89.0s
t79: SP = 20.0, PV = 15.0, MV = 83.3, t = 89.5s
t80: SP = 20.0, PV = 16.1, MV = 83.3, t = 90.0s
t81: SP = 20.0, PV = 17.4, MV = 83.3, t = 90.5s
t82: SP = 20.0, PV = 18.6, MV = 89.8, t = 91.0s
t83: SP = 20.0, PV = 19.7, MV = 89.8, t = 91.5s
t84: SP = 20.0, PV = 20.2, MV = 89.8, t = 92.0s
t85: SP = 20.0, PV = 21.6, MV = 89.8, t = 92.5s
t86: SP = 20.0, PV = 22.7, MV = 85.8, t = 93.0s
t87: SP = 20.0, PV = 23.8, MV = 85.8, t = 93.5s
t88: SP = 20.0, PV = 24.9, MV = 85.8, t = 94.0s
t89: SP = 20.0, PV = 26.0, MV = 85.8, t = 94.5s
t90: SP = 20.0, PV = 27.1, MV = 70.8, t = 95.0s
t91: SP = 20.0, PV = 26.0, MV = 70.8, t = 95.5s
t92: SP = 20.0, PV = 24.9, MV = 70.8, t = 96.0s
t93: SP = 20.0, PV = 23.8, MV = 70.8, t = 96.5s
t94: SP = 20.0, PV = 22.7, MV = 61.3, t = 97.0s
t95: SP = 20.0, PV = 21.6, MV = 61.3, t = 97.5s
t96: SP = 20.0, PV = 20.3, MV = 61.3, t = 98.0s
t97: SP = 20.0, PV = 19.8, MV = 61.3, t = 98.5s
t98: SP = 20.0, PV = 19.7, MV = 61.8, t = 99.0s
t99: SP = 20.0, PV = 18.6, MV = 61.8, t = 99.5s
t100: SP = 20.0, PV = 17.4, MV = 61.8, t = 100.0s
t101: SP = 20.0, PV = 16.1, MV = 61.8, t = 100.5s
t102: SP = 20.0, PV = 15.0, MV = 65.7, t = 101.0s
t103: SP = 20.0, PV = 13.9, MV = 65.7, t = 101.5s
t104: SP = 20.0, PV = 12.8, MV = 65.7, t = 102.0s
t105: SP = 20.0, PV = 13.9, MV = 65.7, t = 102.5s
t106: SP = 20.0, PV = 15.0, MV = 81.0, t = 103.0s
t107: SP = 20.0, PV = 16.1, MV = 81.0, t = 103.5s
t108: SP = 20.0, PV = 17.4, MV = 81.0, t = 104.0s
t109: SP = 20.0, PV = 18.6, MV = 81.0, t = 104.5s
t110: SP = 20.0, PV = 19.7, MV = 81.8, t = 105.0s
t111: SP = 20.0, PV = 20.2, MV = 81.8, t = 105.5s
t112: SP = 20.0, PV = 21.6, MV = 81.8, t = 106.0s
t113: SP = 20.0, PV = 21.2, MV = 81.8, t = 106.5s
t114: SP = 20.0, PV = 21.1, MV = 78.8, t = 107.0s
t115: SP = 20.0, PV = 21.0, MV = 78.8, t = 107.5s
t116: SP = 20.0, PV = 20.7, MV = 78.8, t = 108.0s
t117: SP = 20.0, PV = 20.5, MV = 78.8, t = 108.5s
t118: SP = 20.0, PV = 20.3, MV = 77.6, t = 109.0s
t119: SP = 20.0, PV = 20.2, MV = 77.6, t = 109.5s
t120: SP = 20.0, PV = 20.1, MV = 77.6, t = 110.0s
t121: SP = 20.0, PV = 20.1, MV = 77.6, t = 110.5s
t122: SP = 20.0, PV = 20.1, MV = 77.4, t = 111.0s
t123: SP = 20.0, PV = 20.1, MV = 77.4, t = 111.5s
t124: SP = 20.0, PV = 20.0, MV = 77.4, t = 112.0s
t125: SP = 20.0, PV = 20.0, MV = 77.4, t = 112.5s
t126: SP = 20.0, PV = 20.0, MV = 77.4, t = 113.0s
t127: SP = 20.0, PV = 20.0, MV = 77.4, t = 113.5s
t128: SP = 20.0, PV = 19.9, MV = 77.4, t = 114.0s
t129: SP = 20.0, PV = 19.9, MV = 77.4, t = 114.5s
t130: SP = 20.0, PV = 19.9, MV = 77.4, t = 115.0s
t131: SP = 20.0, PV = 19.8, MV = 77.4, t = 115.5s
t132: SP = 20.0, PV = 19.8, MV = 77.4, t = 116.0s
t133: SP = 20.0, PV = 19.8, MV = 77.4, t = 116.5s
t134: SP = 20.0, PV = 19.8, MV = 77.4, t = 117.0s
t135: SP = 20.0, PV = 19.8, MV = 77.4, t = 117.5s
t136: SP = 20.0, PV = 19.8, MV = 77.4, t = 118.0s
t137: SP = 20.0, PV = 19.8, MV = 77.4, t = 118.5s
t138: SP = 20.0, PV = 19.8, MV = 77.4, t = 119.0s
t139: SP = 20.0, PV = 19.8, MV = 77.4, t = 119.5s
t140: SP = 20.0, PV = 19.8, MV = 77.4, t = 120.0s
t141: SP = 20.0, PV = 19.8, MV = 77.4, t = 120.5s
t142: SP = 20.0, PV = 19.8, MV = 77.4, t = 121.0s

Então, calcula-se o módulo do erro (|erro|) e compara-se o valor de |erro| com o valor de
2% do SP para cada instante de tempo. De acordo com as respectivas variáveis, tem-se:

t0: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t1: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t2: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t3: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t4: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t5: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t6: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t7: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t8: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t9: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t10: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t11: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t12: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t13: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t14: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t15: |erro| = |SP - PV| = |20.0 - 15.7| = | 4.30| = 4.30, (|erro|)>(2% do SP) = (4.30)>(0.4) = True
t16: |erro| = |SP - PV| = |20.0 - 16.5| = | 3.50| = 3.50, (|erro|)>(2% do SP) = (3.50)>(0.4) = True
t17: |erro| = |SP - PV| = |20.0 - 17.3| = | 2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t18: |erro| = |SP - PV| = |20.0 - 18.1| = | 1.90| = 1.90, (|erro|)>(2% do SP) = (1.90)>(0.4) = True
t19: |erro| = |SP - PV| = |20.0 - 18.5| = | 1.50| = 1.50, (|erro|)>(2% do SP) = (1.50)>(0.4) = True
t20: |erro| = |SP - PV| = |20.0 - 18.9| = | 1.10| = 1.10, (|erro|)>(2% do SP) = (1.10)>(0.4) = True
t21: |erro| = |SP - PV| = |20.0 - 19.3| = | 0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t22: |erro| = |SP - PV| = |20.0 - 19.3| = | 0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t23: |erro| = |SP - PV| = |20.0 - 19.4| = | 0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t24: |erro| = |SP - PV| = |20.0 - 19.4| = | 0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t25: |erro| = |SP - PV| = |20.0 - 19.5| = | 0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t26: |erro| = |SP - PV| = |20.0 - 19.5| = | 0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t27: |erro| = |SP - PV| = |20.0 - 19.6| = | 0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t28: |erro| = |SP - PV| = |20.0 - 19.6| = | 0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t29: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t30: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t31: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t32: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t33: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t34: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t35: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t36: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t37: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t38: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t39: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t40: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t41: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t42: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t43: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t44: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t45: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t46: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t47: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t48: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t49: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t50: |erro| = |SP - PV| = |20.0 - 12.8| = | 7.20| = 7.20, (|erro|)>(2% do SP) = (0.00)>(0.4) = True
t51: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t52: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t53: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t54: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t55: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t56: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t57: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t58: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t59: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t60: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t61: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t62: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t63: |erro| = |SP - PV| = |20.0 - 27.1| = |-7.10| = 7.10, (|erro|)>(2% do SP) = (7.10)>(0.4) = True
t64: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t65: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t66: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t67: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t68: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t69: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t70: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t71: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t72: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t73: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t74: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t75: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t76: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t77: |erro| = |SP - PV| = |20.0 - 12.8| = | 7.20| = 7.20, (|erro|)>(2% do SP) = (7.20)>(0.4) = True
t78: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t79: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t80: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t81: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t82: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t83: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t84: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t85: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t86: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t87: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t88: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t89: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t90: |erro| = |SP - PV| = |20.0 - 27.1| = |-7.10| = 7.10, (|erro|)>(2% do SP) = (7.10)>(0.4) = True
t91: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t92: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t93: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t94: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t95: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t96: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t97: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t98: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t99: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t100: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t101: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t102: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t103: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t104: |erro| = |SP - PV| = |20.0 - 12.8| = | 7.20| = 7.20, (|erro|)>(2% do SP) = (7.20)>(0.4) = True
t105: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t106: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t107: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t108: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t109: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t110: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t111: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t112: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t113: |erro| = |SP - PV| = |20.0 - 21.2| = |-1.20| = 1.20, (|erro|)>(2% do SP) = (1.20)>(0.4) = True
t114: |erro| = |SP - PV| = |20.0 - 21.1| = |-1.10| = 1.10, (|erro|)>(2% do SP) = (1.10)>(0.4) = True
t115: |erro| = |SP - PV| = |20.0 - 21.0| = |-1.00| = 1.00, (|erro|)>(2% do SP) = (1.00)>(0.4) = True
t116: |erro| = |SP - PV| = |20.0 - 20.7| = |-0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t117: |erro| = |SP - PV| = |20.0 - 20.5| = |-0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t118: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t119: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t120: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t121: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t122: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t123: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t124: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t125: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t126: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t127: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t128: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t129: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t130: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t131: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t132: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t133: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t134: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t135: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t136: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t137: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t138: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t139: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t140: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t141: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t142: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False

O tempo de amostragem calculado é de Tam = 0.5s e o número de N de instantes de tempo é N = 20 instantes de tempo.

Foram identificados 02 distúrbios.

O instante inicial do distúrbio 01 é em ti1=56.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 01 é em tf1=63.5s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

O instante inicial do distúrbio 02 é em ti2=75.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 02 é em tf2=109.0s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

A Conclusão é que houveram 02 trechos de distúrbio identificados.

O distúrbio 01 ocorre do instante ti1=56.0s a tf1=63.5s.
O distúrbio 02 ocorre do instante ti2=75.0s a tf2=109.0s.

--- Fim do Exemplo 2 de cálculo do instante inicial e final de um distúrbio ---

<==== FIM DA ORIENTAÇÃO TÉCNICA PARA IDENTIFICAÇÃO DE DISTÚRBIOS



====> INÍCIO DAS TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS

Critério C da variação do valor do erro em relação ao SP: Podem existir critérios de para considerar determinadas variações nos valores do módulo do erro (|erro|)
como aceitáveis, que são:

Critério C5%: 5% do valor do SP.
Critério C2%: 2% do valor do SP
Critério C1%: 1% do valor do SP 

--- Zero-Crossings ---

Zero-crossings são situações que ocorrem dentro do distúrbio (entre o instante inicial ti e o instante final tf do distúrbio) que o valor do erro e do |erro| são menores que os valores do critério C
determinado, mas não se mantêm menores do que o critério C determinado por mais que N/3 instantes de tempo.

--- Exemplo 1 de contagem de Zero-Crossings ---

Utilizar o critério C2%.

Sendo as Variáveis:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.7, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.7, t = 57.0s
t15: SP = 20.0, PV = 17.4, MV = 48.7, t = 57.5s
t16: SP = 20.0, PV = 17.4, MV = 48.7, t = 58.0s
t17: SP = 20.0, PV = 17.5, MV = 48.7, t = 58.5s
t18: SP = 20.0, PV = 20.0, MV = 55.0, t = 59.0s
t19: SP = 20.0, PV = 20.0, MV = 55.0, t = 59.5s
t20: SP = 20.0, PV = 20.1, MV = 55.0, t = 60.0s
t21: SP = 20.0, PV = 22.7, MV = 55.0, t = 60.5s
t22: SP = 20.0, PV = 22.7, MV = 55.0, t = 61.0s
t23: SP = 20.0, PV = 22.6, MV = 56.8, t = 61.5s
t24: SP = 20.0, PV = 25.3, MV = 56.8, t = 62.0s
t25: SP = 20.0, PV = 25.3, MV = 56.8, t = 62.5s
t26: SP = 20.0, PV = 25.2, MV = 56.8, t = 63.0s
t27: SP = 20.0, PV = 21.7, MV = 56.8, t = 63.5s
t28: SP = 20.0, PV = 21.7, MV = 57.8, t = 64.0s
t29: SP = 20.0, PV = 21.4, MV = 57.8, t = 64.5s
t30: SP = 20.0, PV = 20.6, MV = 57.8, t = 65.0s
t31: SP = 20.0, PV = 20.6, MV = 57.8, t = 65.5s
t32: SP = 20.0, PV = 20.4, MV = 57.8, t = 66.0s
t33: SP = 20.0, PV = 20.3, MV = 60.0, t = 66.5s
t34: SP = 20.0, PV = 20.2, MV = 60.0, t = 67.0s
t35: SP = 20.0, PV = 20.1, MV = 60.0, t = 67.5s
t36: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.0s
t37: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.5s
t38: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.0s
t39: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.5s
t40: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.0s
t41: SP = 20.0, PV = 20.1, MV = 60.1, t = 70.5s
t42: SP = 20.0, PV = 20.1, MV = 60.1, t = 71.0s
t43: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.5s
t44: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.0s
t45: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.5s
t46: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.0s
t47: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.5s
t48: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.0s
t49: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.5s
t50: SP = 20.0, PV = 20.2, MV = 59.9, t = 75.0s
t51: SP = 20.0, PV = 20.2, MV = 59.9, t = 75.5s
t52: SP = 20.0, PV = 20.1, MV = 59.9, t = 76.0s
t53: SP = 20.0, PV = 20.1, MV = 59.9, t = 76.5s
t54: SP = 20.0, PV = 20.1, MV = 59.9, t = 77.0s
t55: SP = 20.0, PV = 20.1, MV = 59.9, t = 77.5s
t56: SP = 20.0, PV = 20.1, MV = 59.9, t = 78.0s

Então, calcula-se o módulo do erro (|erro|) e compara-se o valor de |erro| com o valor de
2% do SP para cada instante de tempo. De acordo com as respectivas variáveis, tem-se:

t0: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t1: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t2: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t3: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t4: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t5: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t6: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t7: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t8: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t9: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t10: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t11: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t12: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t13: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t14: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t15: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t16: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t17: |erro| = |SP - PV| = |20.0 - 17.5| = | 2.50| = 2.50, (|erro|)>(2% do SP) = (2.50)>(0.4) = True
t18: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t19: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t20: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t21: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t22: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t23: |erro| = |SP - PV| = |20.0 - 22.6| = |-2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t24: |erro| = |SP - PV| = |20.0 - 25.3| = |-5.30| = 5.30, (|erro|)>(2% do SP) = (5.30)>(0.4) = True
t25: |erro| = |SP - PV| = |20.0 - 25.3| = |-5.30| = 5.30, (|erro|)>(2% do SP) = (5.30)>(0.4) = True
t26: |erro| = |SP - PV| = |20.0 - 25.2| = |-5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t27: |erro| = |SP - PV| = |20.0 - 21.7| = |-1.70| = 1.70, (|erro|)>(2% do SP) = (1.70)>(0.4) = True
t28: |erro| = |SP - PV| = |20.0 - 21.7| = |-1.70| = 1.70, (|erro|)>(2% do SP) = (1.70)>(0.4) = True
t29: |erro| = |SP - PV| = |20.0 - 21.4| = |-1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t30: |erro| = |SP - PV| = |20.0 - 20.6| = |-0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t31: |erro| = |SP - PV| = |20.0 - 20.6| = |-0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t32: |erro| = |SP - PV| = |20.0 - 20.4| = |-0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t33: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t34: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t35: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t36: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t37: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t38: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t39: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t40: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t41: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t42: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t43: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t44: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t45: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t46: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t47: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t48: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t49: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t50: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t51: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t52: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t53: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t54: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t55: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t56: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False

O tempo de amostragem calculado é de Tam = 0.5s e o número de N de instantes de tempo é N = 20 instantes de tempo.

Foi identificado 01 distúrbio.

O instante inicial do distúrbio 01 é em ti1=56.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 01 é em tf1=65.5s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

A Conclusão é que houve 01 trecho de distúrbio identificado.

O distúrbio 01 ocorre do instante t=56.0s a t=65.5s.

O número de Zero-crossings para o distúrbio identificado foi de 01 zero-crossings, do instante 59.0s ao instante 60.0s. 

--- Fim do Exemplo 1 de contagem de Zero-Crossings ---


--- Exemplo 2 de contagem de Zero-Crossings ---

Utilizar o critério C2%.

Sendo as Variáveis:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.7, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.7, t = 57.0s
t15: SP = 20.0, PV = 17.4, MV = 48.7, t = 57.5s
t16: SP = 20.0, PV = 17.4, MV = 48.7, t = 58.0s
t17: SP = 20.0, PV = 17.5, MV = 48.7, t = 58.5s
t18: SP = 20.0, PV = 20.0, MV = 55.0, t = 59.0s
t19: SP = 20.0, PV = 20.0, MV = 55.0, t = 59.5s
t20: SP = 20.0, PV = 20.1, MV = 55.0, t = 60.0s
t21: SP = 20.0, PV = 22.7, MV = 55.0, t = 60.5s
t22: SP = 20.0, PV = 22.7, MV = 55.0, t = 61.0s
t23: SP = 20.0, PV = 22.6, MV = 56.8, t = 61.5s
t24: SP = 20.0, PV = 25.3, MV = 56.8, t = 62.0s
t25: SP = 20.0, PV = 25.3, MV = 56.8, t = 62.5s
t26: SP = 20.0, PV = 25.2, MV = 56.8, t = 63.0s
t27: SP = 20.0, PV = 21.7, MV = 56.8, t = 63.5s
t28: SP = 20.0, PV = 21.7, MV = 57.8, t = 64.0s
t29: SP = 20.0, PV = 21.4, MV = 57.8, t = 64.5s
t30: SP = 20.0, PV = 20.6, MV = 57.8, t = 65.0s
t31: SP = 20.0, PV = 20.6, MV = 57.8, t = 65.5s
t32: SP = 20.0, PV = 20.4, MV = 57.8, t = 66.0s
t33: SP = 20.0, PV = 20.3, MV = 60.0, t = 66.5s
t34: SP = 20.0, PV = 20.2, MV = 60.0, t = 67.0s
t35: SP = 20.0, PV = 20.1, MV = 60.0, t = 67.5s
t36: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.0s
t37: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.5s
t38: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.0s
t39: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.5s
t40: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.0s
t41: SP = 20.0, PV = 20.1, MV = 60.1, t = 70.5s
t42: SP = 20.0, PV = 20.1, MV = 60.1, t = 71.0s
t43: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.5s
t44: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.0s
t45: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.5s
t46: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.0s
t47: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.5s
t48: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.0s
t49: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.5s
t50: SP = 20.0, PV = 20.2, MV = 59.9, t = 75.0s
t51: SP = 20.0, PV = 20.2, MV = 59.9, t = 75.5s
t52: SP = 20.0, PV = 20.1, MV = 59.9, t = 76.0s
t53: SP = 20.0, PV = 20.1, MV = 59.9, t = 76.5s
t54: SP = 20.0, PV = 20.1, MV = 59.9, t = 77.0s
t55: SP = 20.0, PV = 20.1, MV = 59.9, t = 77.5s
t56: SP = 20.0, PV = 20.1, MV = 59.9, t = 78.0s

Então, calcula-se o módulo do erro (|erro|) e compara-se o valor de |erro| com o valor de
2% do SP para cada instante de tempo. De acordo com as respectivas variáveis, tem-se:

t0: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t1: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t2: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t3: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t4: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t5: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t6: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t7: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t8: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t9: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t10: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t11: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t12: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t13: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t14: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t15: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t16: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t17: |erro| = |SP - PV| = |20.0 - 17.5| = | 2.50| = 2.50, (|erro|)>(2% do SP) = (2.50)>(0.4) = True
t18: |erro| = |SP - PV| = |20.0 - 20.9| = |-0.90| = 0.90, (|erro|)>(2% do SP) = (0.90)>(0.4) = True
t19: |erro| = |SP - PV| = |20.0 - 20.9| = |-0.90| = 0.90, (|erro|)>(2% do SP) = (0.90)>(0.4) = True
t20: |erro| = |SP - PV| = |20.0 - 21.0| = |-1.00| = 1.00, (|erro|)>(2% do SP) = (1.00)>(0.4) = True
t21: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t22: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t23: |erro| = |SP - PV| = |20.0 - 22.6| = |-2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t24: |erro| = |SP - PV| = |20.0 - 25.3| = |-5.30| = 5.30, (|erro|)>(2% do SP) = (5.30)>(0.4) = True
t25: |erro| = |SP - PV| = |20.0 - 25.3| = |-5.30| = 5.30, (|erro|)>(2% do SP) = (5.30)>(0.4) = True
t26: |erro| = |SP - PV| = |20.0 - 25.2| = |-5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t27: |erro| = |SP - PV| = |20.0 - 21.7| = |-1.70| = 1.70, (|erro|)>(2% do SP) = (1.70)>(0.4) = True
t28: |erro| = |SP - PV| = |20.0 - 21.7| = |-1.70| = 1.70, (|erro|)>(2% do SP) = (1.70)>(0.4) = True
t29: |erro| = |SP - PV| = |20.0 - 21.4| = |-1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t30: |erro| = |SP - PV| = |20.0 - 20.6| = |-0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t31: |erro| = |SP - PV| = |20.0 - 20.6| = |-0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t32: |erro| = |SP - PV| = |20.0 - 20.4| = |-0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t33: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t34: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t35: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t36: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t37: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t38: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t39: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t40: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t41: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t42: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t43: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t44: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t45: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t46: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t47: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t48: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t49: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t50: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t51: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t52: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t53: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t54: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t55: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t56: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False

O tempo de amostragem calculado é de Tam = 0.5s e o número de N de instantes de tempo é N = 20 instantes de tempo.

Foi identificado 01 distúrbio.

O instante inicial do distúrbio 01 é em ti1=56.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 01 é em tf1=65.5s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

A Conclusão é que houve 01 trecho de distúrbio identificado.

O distúrbio 01 ocorre do instante t=56.0s a t=65.5s.

O número de Zero-crossings para o distúrbio identificado foi de 0 zero-crossings, ou seja, não houve ocorrência de zero crossings. 

--- Fim do Exemplo 2 de contagem de Zero-Crossings ---

Os Distúrbios podem ser classificados como transientes, distúrbios isolados e oscilações.

*** Critérios para classificação de Transientes ***

Transiente: é o distúrbio gerado quando ocorre mudança no Set Point (SP) e é gerado um valor de erro significativo no sistema maior que o critério C determinado, ou seja,
quando ocorre uma alteração no valor do SP é gerado um valor de erro (erro = SP - PV) significativo no sistema maior que o critério C determinado e consequentemente um valor de |erro| (|erro| = |SP - PV|)
significativo no sistema maior que o critério C determinado, seguido de uma ação na MV para que seja ajustada a PV e corrigido o valor do erro e |erro| para valores menores que os valores do critério C determinado,
e em seguida os valores do |erro| permanecem menores que os valores do critério C determinado por N instantes de tempo após o tempo final tf do distúrbio. Nesse caso o trecho do distúrbio é classificado como Transiente.

Para identificar se ocorreu um transiente em um trecho, observar os seguintes aspectos:

- Obrigatoriamente deve existir alteração do SP em instantes próximos ao instante inicial ti do distúrbio. Caso não exista alteração do SP em instantes próximos ao instante inicial ti do distúrbio, não é um transiente.
- Transientes possuem um número de zero-crossings menor que 2.


*** Critérios para classificação de Distúrbios Isolados ***

Distúrbio Isolado: é quando ocorre um valor de erro (erro = SP - PV) maior que o critério C determinado e consequente |erro| (|erro| = |SP - PV|) maior que o critério C determinado sem ocorrência de alteração do SP,
ou seja, valor de erro (erro = SP - PV) e do |erro| (|erro| = |SP - PV|) são maiores que o critério C determinado sem ter ocorrido uma alteração no SP. Este evento é seguido de uma ação na MV para que seja ajustada
a PV e corrigido o valor do erro e |erro| para valores menores que os valores do critério C determinado, e em seguida os valores do |erro| permanecem menores que os valores do critério C determinado por N instantes
de tempo após o tempo final tf do distúrbio. Nesse caso o trecho do distúrbio é classificado como Distúrbio Isolado.

Para identificar se ocorreu um distúrbio isolado em um trecho, observar os seguintes aspectos:

- Obrigatoriamente não deve existir alteração do SP em instantes próximos ao instante inicial ti do distúrbio. Caso exista alteração do SP em instantes próximos ao instante inicial ti do distúrbio, não é um distúrbio isolado.
- Distúrbios isolados possuem um número de zero-crossings menor que 2.


*** Critérios para classificação de Oscilações ***

Oscilação: é quando ocorre um valor de erro entre o SP e a PV sem alteração do SP, seguido de uma ação na MV para que seja ajustada a PV, mas ocorre a oscilação do valor da PV, com o valor do erro somente
passando pelo valor de zero em um pequeno instante de tempo, sem ocorrer estabilidade no sistema de controle no trecho.
A estabilidade do sistema é definida como N instantes de tempo após o tempo final tf da oscilação.

Para identificar se ocorreu uma oscilação em um trecho, observar os seguintes aspectos:

- Oscilações podem ocorrer tanto quando ocorrer alteração do SP, quanto quando não ocorrer alteração do SP.
- Oscilações possuem um número de zero-crossings maior que 2.


--- Exemplo 1 de classificação de distúrbios ---

Utilizar o critério C2%.

Sendo as Variáveis:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.7, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.7, t = 57.0s
t15: SP = 20.0, PV = 15.7, MV = 48.7, t = 57.5s
t16: SP = 20.0, PV = 16.5, MV = 48.7, t = 58.0s
t17: SP = 20.0, PV = 17.3, MV = 48.7, t = 58.5s
t18: SP = 20.0, PV = 18.1, MV = 55.0, t = 59.0s
t19: SP = 20.0, PV = 18.5, MV = 55.0, t = 59.5s
t20: SP = 20.0, PV = 18.9, MV = 55.0, t = 60.0s
t21: SP = 20.0, PV = 19.3, MV = 55.0, t = 60.5s
t22: SP = 20.0, PV = 19.3, MV = 55.0, t = 61.0s
t23: SP = 20.0, PV = 19.4, MV = 56.8, t = 61.5s
t24: SP = 20.0, PV = 19.4, MV = 56.8, t = 62.0s
t25: SP = 20.0, PV = 19.5, MV = 56.8, t = 62.5s
t26: SP = 20.0, PV = 19.5, MV = 56.8, t = 63.0s
t27: SP = 20.0, PV = 19.6, MV = 56.8, t = 63.5s
t28: SP = 20.0, PV = 19.6, MV = 57.8, t = 64.0s
t29: SP = 20.0, PV = 19.7, MV = 57.8, t = 64.5s
t30: SP = 20.0, PV = 19.7, MV = 57.8, t = 65.0s
t31: SP = 20.0, PV = 19.8, MV = 57.8, t = 65.5s
t32: SP = 20.0, PV = 19.8, MV = 57.8, t = 66.0s
t33: SP = 20.0, PV = 19.8, MV = 60.0, t = 66.5s
t34: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.0s
t35: SP = 20.0, PV = 19.9, MV = 60.0, t = 67.5s
t36: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.0s
t37: SP = 20.0, PV = 19.9, MV = 60.0, t = 68.5s
t38: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.0s
t39: SP = 20.0, PV = 20.0, MV = 60.1, t = 69.5s
t40: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.0s
t41: SP = 20.0, PV = 20.0, MV = 60.1, t = 70.5s
t42: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.0s
t43: SP = 20.0, PV = 20.1, MV = 60.0, t = 71.5s
t44: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.0s
t45: SP = 20.0, PV = 20.1, MV = 60.0, t = 72.5s
t46: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.0s
t47: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.5s
t48: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.0s
t49: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.5s
t50: SP = 20.0, PV = 12.8, MV = 78.1, t = 75.0s
t51: SP = 20.0, PV = 13.9, MV = 78.1, t = 75.5s
t52: SP = 20.0, PV = 15.0, MV = 78.1, t = 76.0s
t53: SP = 20.0, PV = 16.1, MV = 78.1, t = 76.5s
t54: SP = 20.0, PV = 17.4, MV = 84.6, t = 77.0s
t55: SP = 20.0, PV = 18.6, MV = 84.6, t = 77.5s
t56: SP = 20.0, PV = 19.7, MV = 84.6, t = 78.0s
t57: SP = 20.0, PV = 20.2, MV = 84.6, t = 78.5s
t58: SP = 20.0, PV = 21.6, MV = 84.0, t = 79.0s
t59: SP = 20.0, PV = 22.7, MV = 84.0, t = 79.5s
t60: SP = 20.0, PV = 23.8, MV = 84.0, t = 80.0s
t61: SP = 20.0, PV = 24.9, MV = 84.0, t = 80.5s
t62: SP = 20.0, PV = 26.0, MV = 71.8, t = 81.0s
t63: SP = 20.0, PV = 27.1, MV = 71.8, t = 81.5s
t64: SP = 20.0, PV = 26.0, MV = 71.8, t = 82.0s
t65: SP = 20.0, PV = 24.9, MV = 71.8, t = 82.5s
t66: SP = 20.0, PV = 23.8, MV = 59.6, t = 83.0s
t67: SP = 20.0, PV = 22.7, MV = 59.6, t = 83.5s
t68: SP = 20.0, PV = 21.6, MV = 59.6, t = 84.0s
t69: SP = 20.0, PV = 20.3, MV = 59.6, t = 84.5s
t70: SP = 20.0, PV = 19.8, MV = 58.8, t = 85.0s
t71: SP = 20.0, PV = 19.7, MV = 58.8, t = 85.5s
t72: SP = 20.0, PV = 18.6, MV = 58.8, t = 86.0s
t73: SP = 20.0, PV = 17.4, MV = 58.8, t = 86.5s
t74: SP = 20.0, PV = 16.1, MV = 65.3, t = 87.0s
t75: SP = 20.0, PV = 15.0, MV = 65.3, t = 87.5s
t76: SP = 20.0, PV = 13.9, MV = 65.3, t = 88.0s
t77: SP = 20.0, PV = 12.8, MV = 65.3, t = 88.5s
t78: SP = 20.0, PV = 13.9, MV = 83.3, t = 89.0s
t79: SP = 20.0, PV = 15.0, MV = 83.3, t = 89.5s
t80: SP = 20.0, PV = 16.1, MV = 83.3, t = 90.0s
t81: SP = 20.0, PV = 17.4, MV = 83.3, t = 90.5s
t82: SP = 20.0, PV = 18.6, MV = 89.8, t = 91.0s
t83: SP = 20.0, PV = 19.7, MV = 89.8, t = 91.5s
t84: SP = 20.0, PV = 20.2, MV = 89.8, t = 92.0s
t85: SP = 20.0, PV = 21.6, MV = 89.8, t = 92.5s
t86: SP = 20.0, PV = 22.7, MV = 85.8, t = 93.0s
t87: SP = 20.0, PV = 23.8, MV = 85.8, t = 93.5s
t88: SP = 20.0, PV = 24.9, MV = 85.8, t = 94.0s
t89: SP = 20.0, PV = 26.0, MV = 85.8, t = 94.5s
t90: SP = 20.0, PV = 27.1, MV = 70.8, t = 95.0s
t91: SP = 20.0, PV = 26.0, MV = 70.8, t = 95.5s
t92: SP = 20.0, PV = 24.9, MV = 70.8, t = 96.0s
t93: SP = 20.0, PV = 23.8, MV = 70.8, t = 96.5s
t94: SP = 20.0, PV = 22.7, MV = 61.3, t = 97.0s
t95: SP = 20.0, PV = 21.6, MV = 61.3, t = 97.5s
t96: SP = 20.0, PV = 20.3, MV = 61.3, t = 98.0s
t97: SP = 20.0, PV = 19.8, MV = 61.3, t = 98.5s
t98: SP = 20.0, PV = 19.7, MV = 61.8, t = 99.0s
t99: SP = 20.0, PV = 18.6, MV = 61.8, t = 99.5s
t100: SP = 20.0, PV = 17.4, MV = 61.8, t = 100.0s
t101: SP = 20.0, PV = 16.1, MV = 61.8, t = 100.5s
t102: SP = 20.0, PV = 15.0, MV = 65.7, t = 101.0s
t103: SP = 20.0, PV = 13.9, MV = 65.7, t = 101.5s
t104: SP = 20.0, PV = 12.8, MV = 65.7, t = 102.0s
t105: SP = 20.0, PV = 13.9, MV = 65.7, t = 102.5s
t106: SP = 20.0, PV = 15.0, MV = 81.0, t = 103.0s
t107: SP = 20.0, PV = 16.1, MV = 81.0, t = 103.5s
t108: SP = 20.0, PV = 17.4, MV = 81.0, t = 104.0s
t109: SP = 20.0, PV = 18.6, MV = 81.0, t = 104.5s
t110: SP = 20.0, PV = 19.7, MV = 81.8, t = 105.0s
t111: SP = 20.0, PV = 20.2, MV = 81.8, t = 105.5s
t112: SP = 20.0, PV = 21.6, MV = 81.8, t = 106.0s
t113: SP = 20.0, PV = 21.2, MV = 81.8, t = 106.5s
t114: SP = 20.0, PV = 21.1, MV = 78.8, t = 107.0s
t115: SP = 20.0, PV = 21.0, MV = 78.8, t = 107.5s
t116: SP = 20.0, PV = 20.7, MV = 78.8, t = 108.0s
t117: SP = 20.0, PV = 20.5, MV = 78.8, t = 108.5s
t118: SP = 20.0, PV = 20.3, MV = 77.6, t = 109.0s
t119: SP = 20.0, PV = 20.2, MV = 77.6, t = 109.5s
t120: SP = 20.0, PV = 20.1, MV = 77.6, t = 110.0s
t121: SP = 20.0, PV = 20.1, MV = 77.6, t = 110.5s
t122: SP = 20.0, PV = 20.1, MV = 77.4, t = 111.0s
t123: SP = 20.0, PV = 20.1, MV = 77.4, t = 111.5s
t124: SP = 20.0, PV = 20.0, MV = 77.4, t = 112.0s
t125: SP = 20.0, PV = 20.0, MV = 77.4, t = 112.5s
t126: SP = 20.0, PV = 20.0, MV = 77.4, t = 113.0s
t127: SP = 20.0, PV = 20.0, MV = 77.4, t = 113.5s
t128: SP = 20.0, PV = 19.9, MV = 77.4, t = 114.0s
t129: SP = 20.0, PV = 19.9, MV = 77.4, t = 114.5s
t130: SP = 20.0, PV = 19.9, MV = 77.4, t = 115.0s
t131: SP = 20.0, PV = 19.8, MV = 77.4, t = 115.5s
t132: SP = 20.0, PV = 19.8, MV = 77.4, t = 116.0s
t133: SP = 20.0, PV = 19.8, MV = 77.4, t = 116.5s
t134: SP = 20.0, PV = 19.8, MV = 77.4, t = 117.0s
t135: SP = 20.0, PV = 19.8, MV = 77.4, t = 117.5s
t136: SP = 20.0, PV = 19.8, MV = 77.4, t = 118.0s
t137: SP = 20.0, PV = 19.8, MV = 77.4, t = 118.5s
t138: SP = 20.0, PV = 19.8, MV = 77.4, t = 119.0s
t139: SP = 20.0, PV = 19.8, MV = 77.4, t = 119.5s
t140: SP = 20.0, PV = 19.8, MV = 77.4, t = 120.0s
t141: SP = 20.0, PV = 19.8, MV = 77.4, t = 120.5s
t142: SP = 20.0, PV = 19.8, MV = 77.4, t = 121.0s

Então, calcula-se o módulo do erro (|erro|) e compara-se o valor de |erro| com o valor de
2% do SP para cada instante de tempo. De acordo com as respectivas variáveis, tem-se:

t0: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t1: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t2: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t3: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t4: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t5: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t6: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t7: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t8: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t9: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t10: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t11: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t12: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t13: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t14: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t15: |erro| = |SP - PV| = |20.0 - 15.7| = | 4.30| = 4.30, (|erro|)>(2% do SP) = (4.30)>(0.4) = True
t16: |erro| = |SP - PV| = |20.0 - 16.5| = | 3.50| = 3.50, (|erro|)>(2% do SP) = (3.50)>(0.4) = True
t17: |erro| = |SP - PV| = |20.0 - 17.3| = | 2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t18: |erro| = |SP - PV| = |20.0 - 18.1| = | 1.90| = 1.90, (|erro|)>(2% do SP) = (1.90)>(0.4) = True
t19: |erro| = |SP - PV| = |20.0 - 18.5| = | 1.50| = 1.50, (|erro|)>(2% do SP) = (1.50)>(0.4) = True
t20: |erro| = |SP - PV| = |20.0 - 18.9| = | 1.10| = 1.10, (|erro|)>(2% do SP) = (1.10)>(0.4) = True
t21: |erro| = |SP - PV| = |20.0 - 19.3| = | 0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t22: |erro| = |SP - PV| = |20.0 - 19.3| = | 0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t23: |erro| = |SP - PV| = |20.0 - 19.4| = | 0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t24: |erro| = |SP - PV| = |20.0 - 19.4| = | 0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t25: |erro| = |SP - PV| = |20.0 - 19.5| = | 0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t26: |erro| = |SP - PV| = |20.0 - 19.5| = | 0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t27: |erro| = |SP - PV| = |20.0 - 19.6| = | 0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t28: |erro| = |SP - PV| = |20.0 - 19.6| = | 0.40| = 0.40, (|erro|)>(2% do SP) = (0.40)>(0.4) = False
t29: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t30: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t31: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t32: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t33: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t34: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t35: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t36: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t37: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t38: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t39: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t40: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t41: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t42: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t43: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t44: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t45: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t46: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t47: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t48: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t49: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t50: |erro| = |SP - PV| = |20.0 - 12.8| = | 7.20| = 7.20, (|erro|)>(2% do SP) = (0.00)>(0.4) = True
t51: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t52: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t53: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t54: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t55: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t56: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t57: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t58: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t59: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t60: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t61: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t62: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t63: |erro| = |SP - PV| = |20.0 - 27.1| = |-7.10| = 7.10, (|erro|)>(2% do SP) = (7.10)>(0.4) = True
t64: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t65: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t66: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t67: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t68: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t69: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t70: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t71: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t72: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t73: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t74: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t75: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t76: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t77: |erro| = |SP - PV| = |20.0 - 12.8| = | 7.20| = 7.20, (|erro|)>(2% do SP) = (7.20)>(0.4) = True
t78: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t79: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t80: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t81: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t82: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t83: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t84: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t85: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t86: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t87: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t88: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t89: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t90: |erro| = |SP - PV| = |20.0 - 27.1| = |-7.10| = 7.10, (|erro|)>(2% do SP) = (7.10)>(0.4) = True
t91: |erro| = |SP - PV| = |20.0 - 26.0| = |-6.00| = 6.00, (|erro|)>(2% do SP) = (6.00)>(0.4) = True
t92: |erro| = |SP - PV| = |20.0 - 24.9| = |-4.90| = 4.90, (|erro|)>(2% do SP) = (4.90)>(0.4) = True
t93: |erro| = |SP - PV| = |20.0 - 23.8| = |-3.80| = 3.80, (|erro|)>(2% do SP) = (3.80)>(0.4) = True
t94: |erro| = |SP - PV| = |20.0 - 22.7| = |-2.70| = 2.70, (|erro|)>(2% do SP) = (2.70)>(0.4) = True
t95: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t96: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t97: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t98: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t99: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t100: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t101: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t102: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t103: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t104: |erro| = |SP - PV| = |20.0 - 12.8| = | 7.20| = 7.20, (|erro|)>(2% do SP) = (7.20)>(0.4) = True
t105: |erro| = |SP - PV| = |20.0 - 13.9| = | 6.10| = 6.10, (|erro|)>(2% do SP) = (6.10)>(0.4) = True
t106: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t107: |erro| = |SP - PV| = |20.0 - 16.1| = | 3.90| = 3.90, (|erro|)>(2% do SP) = (3.90)>(0.4) = True
t108: |erro| = |SP - PV| = |20.0 - 17.4| = | 2.60| = 2.60, (|erro|)>(2% do SP) = (2.60)>(0.4) = True
t109: |erro| = |SP - PV| = |20.0 - 18.6| = | 1.40| = 1.40, (|erro|)>(2% do SP) = (1.40)>(0.4) = True
t110: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t111: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t112: |erro| = |SP - PV| = |20.0 - 21.6| = |-1.60| = 1.60, (|erro|)>(2% do SP) = (1.60)>(0.4) = True
t113: |erro| = |SP - PV| = |20.0 - 21.2| = |-1.20| = 1.20, (|erro|)>(2% do SP) = (1.20)>(0.4) = True
t114: |erro| = |SP - PV| = |20.0 - 21.1| = |-1.10| = 1.10, (|erro|)>(2% do SP) = (1.10)>(0.4) = True
t115: |erro| = |SP - PV| = |20.0 - 21.0| = |-1.00| = 1.00, (|erro|)>(2% do SP) = (1.00)>(0.4) = True
t116: |erro| = |SP - PV| = |20.0 - 20.7| = |-0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.4) = True
t117: |erro| = |SP - PV| = |20.0 - 20.5| = |-0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.4) = True
t118: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t119: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t120: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t121: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t122: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t123: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t124: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t125: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t126: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t127: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t128: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t129: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t130: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t131: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t132: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t133: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t134: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t135: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t136: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t137: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t138: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t139: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t140: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t141: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t142: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False

O tempo de amostragem calculado é de Tam = 0.5s e o número de N de instantes de tempo é N = 20 instantes de tempo.

Foram identificados 02 distúrbios.

O instante inicial do distúrbio 01 é em ti1=56.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 01 é em tf1=63.5s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

O instante inicial do distúrbio 02 é em ti2=75.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 02 é em tf2=109.0s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

Houveram 02 trechos de distúrbio identificados.

O distúrbio 01 ocorre do instante ti1=56.0s a tf1=63.5s.
O distúrbio 02 ocorre do instante ti2=75.0s a tf2=109.0s.

Com os distúrbios devidamente identificados, realiza-se a classificação dos distúrbios:

O distúrbio 01 ocorre do instante ti1=56.0s a tf1=63.5s, é um distúrbio isolado pois ocorre um valor de erro entre o SP e a PV sem alteração do SP, seguido
de uma ação na MV para que seja ajustada a PV, e em seguida ocorrem N = 20 instantes de tempo com valores de |erro| menores do que o critério C2%, e não houve ocorrência de zero-crossings.

O distúrbio 02 ocorre do instante ti2=75.0s a tf2=109.0s, é uma oscilação pois um valor de erro entre o SP e a PV sem alteração do SP, seguido de uma ação na MV para que seja ajustada a PV, e ocorrem 
05 zero-crossings, nos instantes de 78.0s a 70.5s, de 84.5s a 85.5s, de 91.5s a 92.0s, de 98.0s a 99.s e de 105.0 a 105.5s, o que permite concluir que é uma oscilação que possui 5 zero-crossings.

--- Fim do Exemplo 1 de classificação de distúrbios ---

<==== FIM DAS TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS



====> INÍCIO DAS TÉCNICAS PARA AVALIAÇÃO DE AGARRAMENTO EM VÁLVULAS DE CONTROLE

Critério C da variação do valor do erro em relação ao SP: Podem existir critérios de para considerar determinadas variações nos valores do módulo do erro (|erro|)
como aceitáveis, que são:

Critério C5%: 5% do valor do SP
Critério C2%: 2% do valor do SP
Critério C1%: 1% do valor do SP 

Tempo de reação da MV e da PV: é o tempo em que a MV e a PV reagem na ocorrência de um distúrbio, sendo:

TrPV: Tempo de Reação da PV
TrMV: Tempo de Reação da MV

Tam: Tempo de Amostragem

--- Agarramento em Distúrbios transientes ---

O agarramento pode apresentar comportamentos específicos durante o um distúrbio de transiente, como a defasagem total ou parcial
entre a reação do sinal de MV e a reação do sinal da PV quando ocorre uma alteração no valor do SP, ou seja, quando ocorre um
distúrbio do tipo transiente, ocorre mudança no Set Point (SP) e é gerado um valor de erro significativo no sistema maior que o
critério C determinado, e ocorre também um grande atraso no tempo de reação da PV em relação ao tempo de reação da MV,
então esta situação é classificada como agarramento em um transiente.

Caso o tempo de reação da PV seja maior que (1.5/Tam) vezes o tempo de reação da MV, é considerado que há presença de agarramento no transiente.
O tempo de reação aceitável da PV deve ser menor ou igual a (1.5/Tam) vezes o tempo de reação da MV para que não seja considerado agarramento.

(TrPV) <= ((1.5/Tam)*TrMV) = True, então não há evidência de agarramento no distúrbio transiente.
(TrPV) <= ((1.5/Tam)*TrMV) = False, então há evidência de agarramento no distúrbio transiente.

Para identificar se ocorreu um agarramento em um transiente em um trecho, observar os seguintes aspectos:

- Obrigatoriamente deve existir alteração do SP em instantes próximos ao instante inicial ti do distúrbio. Caso não exista alteração do SP
em instantes próximos ao instante inicial ti do distúrbio, não é um transiente.
- Transientes possuem um número de zero-crossings menor que 2.
- Tempo de reação da PV (TrPV) maior ou igual a (1.5/Tam) vezes o tempo de reação da MV (TrMV).

--- Exemplo 1 de comportamento específico de agarramento em um transiente ---

Utilizar o critério C2%.

Sendo as Variáveis:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 40.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 40.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 40.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 40.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 40.0, PV = 19.8, MV = 35.5, t = 56.0s
t13: SP = 40.0, PV = 19.8, MV = 48.5, t = 56.5s
t14: SP = 40.0, PV = 19.9, MV = 48.5, t = 57.0s
t15: SP = 40.0, PV = 19.9, MV = 48.5, t = 57.5s
t16: SP = 40.0, PV = 19.8, MV = 48.5, t = 58.0s
t17: SP = 40.0, PV = 19.8, MV = 48.5, t = 58.5s
t18: SP = 40.0, PV = 19.9, MV = 61.3, t = 59.0s
t19: SP = 40.0, PV = 19.9, MV = 61.3, t = 59.5s
t20: SP = 40.0, PV = 19.8, MV = 61.3, t = 60.0s
t21: SP = 40.0, PV = 19.8, MV = 61.3, t = 60.5s
t22: SP = 40.0, PV = 19.9, MV = 61.3, t = 61.0s
t23: SP = 40.0, PV = 19.9, MV = 74.2, t = 61.5s
t24: SP = 40.0, PV = 19.8, MV = 74.2, t = 62.0s
t25: SP = 40.0, PV = 19.8, MV = 74.2, t = 62.5s
t26: SP = 40.0, PV = 26.2, MV = 74.2, t = 63.0s
t27: SP = 40.0, PV = 26.2, MV = 87.3, t = 63.5s
t28: SP = 40.0, PV = 33.7, MV = 87.3, t = 64.0s
t29: SP = 40.0, PV = 33.7, MV = 87.3, t = 64.5s
t30: SP = 40.0, PV = 35.9, MV = 87.3, t = 65.0s
t31: SP = 40.0, PV = 35.9, MV = 100.0, t = 65.5s
t32: SP = 40.0, PV = 39.9, MV = 100.0, t = 66.0s
t33: SP = 40.0, PV = 39.9, MV = 100.0, t = 66.5s
t34: SP = 40.0, PV = 45.5, MV = 100.0, t = 67.0s
t35: SP = 40.0, PV = 45.5, MV = 100.0, t = 67.5s
t36: SP = 40.0, PV = 47.2, MV = 100.0, t = 68.0s
t37: SP = 40.0, PV = 47.2, MV = 100.0, t = 68.5s
t38: SP = 40.0, PV = 44.2, MV = 100.0, t = 69.0s
t39: SP = 40.0, PV = 44.2, MV = 65.7, t = 69.5s
t40: SP = 40.0, PV = 42.2, MV = 65.7, t = 70.0s
t41: SP = 40.0, PV = 42.2, MV = 65.7, t = 70.5s
t42: SP = 40.0, PV = 40.7, MV = 65.7, t = 71.0s
t43: SP = 40.0, PV = 40.5, MV = 59.9, t = 71.5s
t44: SP = 40.0, PV = 40.2, MV = 59.9, t = 72.0s
t45: SP = 40.0, PV = 40.2, MV = 59.9, t = 72.5s
t46: SP = 40.0, PV = 40.2, MV = 59.9, t = 73.0s
t47: SP = 40.0, PV = 40.2, MV = 59.9, t = 73.5s
t48: SP = 40.0, PV = 40.2, MV = 59.9, t = 74.0s
t49: SP = 40.0, PV = 40.2, MV = 59.9, t = 74.5s
t50: SP = 40.0, PV = 40.2, MV = 59.9, t = 75.0s
t51: SP = 40.0, PV = 40.2, MV = 59.9, t = 75.5s
t52: SP = 40.0, PV = 40.2, MV = 59.9, t = 76.0s
t53: SP = 40.0, PV = 40.2, MV = 59.9, t = 76.5s
t54: SP = 40.0, PV = 40.1, MV = 59.9, t = 77.0s
t55: SP = 40.0, PV = 40.1, MV = 59.9, t = 77.5s
t56: SP = 40.0, PV = 40.1, MV = 59.9, t = 78.0s
t57: SP = 40.0, PV = 40.1, MV = 59.9, t = 78.5s
t58: SP = 40.0, PV = 40.0, MV = 59.9, t = 79.0s
t59: SP = 40.0, PV = 40.0, MV = 59.9, t = 79.5s
t60: SP = 40.0, PV = 40.0, MV = 59.9, t = 80.0s
t61: SP = 40.0, PV = 40.0, MV = 59.9, t = 80.5s
t62: SP = 40.0, PV = 40.0, MV = 59.9, t = 81.0s
t63: SP = 40.0, PV = 40.1, MV = 59.9, t = 81.5s
t64: SP = 40.0, PV = 40.1, MV = 59.9, t = 82.0s
t65: SP = 40.0, PV = 40.1, MV = 59.9, t = 82.5s
t66: SP = 40.0, PV = 40.1, MV = 59.9, t = 83.0s

Então, calcula-se o módulo do erro (|erro|) e compara-se o valor de |erro| com o valor de
2% do SP para cada instante de tempo. Caso o valor do |erro| seja maior que 2% do SP, então avalia-se também
o tempo de reação da MV e o tempo de reação da PV para verificar se há indícios de agarramento durante o intervalo ti e tf do distúrbio.
De acordo com as respectivas variáveis, tem-se:

t0: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t1: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t2: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t3: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t4: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t5: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t6: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t7: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t8: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t9: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.20)>(0.8) = True
t10: |erro| = |SP - PV| = |40.0 - 19.7| = | 20.30| = 20.30, (|erro|)>(2% do SP) = (20.30)>(0.8) = True
t11: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t12: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.20)>(0.8) = True
t13: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.20)>(0.8) = True
t14: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t15: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t16: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.20)>(0.8) = True
t17: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.20)>(0.8) = True
t18: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t19: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t20: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.20)>(0.8) = True
t21: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.200)>(0.8) = True
t22: |erro| = |SP - PV| = |40.0 - 19.8| = | 20.20| = 20.20, (|erro|)>(2% do SP) = (20.20)>(0.8) = True
t23: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t24: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t25: |erro| = |SP - PV| = |40.0 - 19.9| = | 20.10| = 20.10, (|erro|)>(2% do SP) = (20.10)>(0.8) = True
t26: |erro| = |SP - PV| = |40.0 - 26.2| = | 13.80| = 13.80, (|erro|)>(2% do SP) = (13.80)>(0.8) = True
t27: |erro| = |SP - PV| = |40.0 - 26.2| = | 13.80| = 13.80, (|erro|)>(2% do SP) = (13.80)>(0.8) = True
t28: |erro| = |SP - PV| = |40.0 - 33.7| = | 6.30| = 6.30, (|erro|)>(2% do SP) = (6.30)>(0.8) = True
t29: |erro| = |SP - PV| = |40.0 - 33.7| = | 6.30| = 6.30, (|erro|)>(2% do SP) = (6.30)>(0.8) = True
t30: |erro| = |SP - PV| = |40.0 - 35.9| = | 4.10| = 4.10, (|erro|)>(2% do SP) = (4.10)>(0.8) = True
t31: |erro| = |SP - PV| = |40.0 - 35.9| = | 4.10| = 4.10, (|erro|)>(2% do SP) = (4.10)>(0.8) = True
t32: |erro| = |SP - PV| = |40.0 - 39.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t33: |erro| = |SP - PV| = |40.0 - 39.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t34: |erro| = |SP - PV| = |40.0 - 45.5| = |-5.50| = 5.50, (|erro|)>(2% do SP) = (5.50)>(0.8) = True
t35: |erro| = |SP - PV| = |40.0 - 45.5| = |-5.50| = 5.50, (|erro|)>(2% do SP) = (5.50)>(0.8) = True
t36: |erro| = |SP - PV| = |40.0 - 47.2| = |-7.20| = 7.20, (|erro|)>(2% do SP) = (7.20)>(0.8) = True
t37: |erro| = |SP - PV| = |40.0 - 47.2| = |-7.20| = 7.20, (|erro|)>(2% do SP) = (7.20)>(0.8) = True
t38: |erro| = |SP - PV| = |40.0 - 44.2| = |-4.20| = 4.20, (|erro|)>(2% do SP) = (4.20)>(0.8) = True
t39: |erro| = |SP - PV| = |40.0 - 44.2| = |-4.20| = 4.20, (|erro|)>(2% do SP) = (4.20)>(0.8) = True
t40: |erro| = |SP - PV| = |40.0 - 42.2| = |-2.20| = 2.20, (|erro|)>(2% do SP) = (2.20)>(0.8) = True
t41: |erro| = |SP - PV| = |40.0 - 42.2| = |-2.20| = 2.20, (|erro|)>(2% do SP) = (2.20)>(0.8) = True
t42: |erro| = |SP - PV| = |40.0 - 40.7| = |-0.70| = 0.70, (|erro|)>(2% do SP) = (0.70)>(0.8) = False
t43: |erro| = |SP - PV| = |40.0 - 40.5| = |-0.50| = 0.50, (|erro|)>(2% do SP) = (0.50)>(0.8) = False
t44: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t45: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t46: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t47: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t48: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t49: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t50: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t51: |erro| = |SP - PV| = |40.0 - 40.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.8) = False
t52: |erro| = |SP - PV| = |40.0 - 40.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t53: |erro| = |SP - PV| = |40.0 - 40.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t54: |erro| = |SP - PV| = |40.0 - 40.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t55: |erro| = |SP - PV| = |40.0 - 40.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t56: |erro| = |SP - PV| = |40.0 - 40.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t57: |erro| = |SP - PV| = |40.0 - 40.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.8) = False
t58: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t59: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t60: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t61: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t62: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t63: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t64: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t65: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False
t66: |erro| = |SP - PV| = |40.0 - 40.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.8) = False

O tempo de amostragem calculado é de Tam = 0.5s e o número de N de instantes de tempo é N = 20 instantes de tempo.

Foi identificado 01 distúrbio.

O instante inicial do distúrbio 01 é em ti1=54.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 01 é em tf1=71.0s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

O distúrbio 01 ocorre do instante t=54.0s a t=71.0s.

O número de Zero-crossings para o distúrbio identificado foi de 01 zero-crossings, no instante 66.0s a 66.5s. 

Com o distúrbio devidamente identificado, realiza-se a classificação do distúrbio:

O distúrbio 01 ocorre do instante ti1=54.0s a tf1=71.0s, é um distúrbio transiente pois ocorre um valor de erro entre o SP e a PV devido a alteração do SP ( nos instantes t=53.5s e t=54.0s), seguido
de uma ação na MV (instante t=56.5s) para que seja ajustada a PV, e após instante t=71.0s (tf1=71.0s) ocorrem N = 20 instantes de tempo com valores de |erro| menores do que o critério C2%, com
ocorrência de 01 zero-crossings, no instante 66.0s a 66.5s.

Então, o distúrbio 01 é classificado como Distúrbio Transiente.

Com o distúrbio devidamente identificado e classificado, verifica-se se há evidências de agarramento:

Dentro do intervalo t=54.0s a t=71.0s do distúrbio 01 identificado, o instante da primeira reação da MV foi em t=56.5s, portanto o TrMV=2.5s, pois a MV reagiu 2.5 segundos
após o início do distúrbio. ti1=54.0s. O instante da primeira reação da PV foi em t=63.0s, portanto o TrPV=9.0s, pois a PV reagiu 9.0 segundos após o início do distúrbio ti1=54.0s.

Então:

TrPV = 9.0s
TrMV = 2.5s
E o tempo de amostragem Tam = 0.5

(TrPV) <= ((1.5/Tam)*TrMV) = (9.0)<=((1.5/0.5)*2.5) = (9.0)<=(7.5) = False. Então, há indícios de agarramento.

A Conclusão é que houve 01 trecho de distúrbio identificado do instante t=54.0s a t=71.0s, classificado como distúrbio Transiente e há evidência de agarramento.

--- Fim do Exemplo 1 de comportamento específico de agarramento em transiente ---


--- Agarramento em Distúrbios isolados ---

O agarramento pode apresentar comportamentos específicos durante o um distúrbio isolado, como a defasagem total ou parcial
entre a reação do sinal de MV e a reação do sinal da PV sem a ocorrência de uma alteração no valor do SP, ou seja, quando ocorre um distúrbio do tipo distúrbio isolado, é gerado um
valor de erro significativo no sistema maior que o critério C determinado sem ocorrer mudança no Set Point (SP), e ocorre também um grande atraso no tempo de reação da PV em relação
ao tempo de reação da MV, então esta situação é classificada como agarramento em um distúrbio isolado.

Caso o tempo de reação da PV seja maior que (1.5/Tam) vezes o tempo de reação da MV, é considerado que há presença de agarramento no distúrbio isolado.
O tempo de reação aceitável da PV deve ser menor ou igual a (1.5/Tam) vezes o tempo de reação da MV para que não seja considerado agarramento.

(TrPV) <= ((1.5/Tam)*TrMV) = True, então não há evidência de agarramento no distúrbio isolado.
(TrPV) <= ((1.5/Tam)*TrMV) = False, então há evidência de agarramento no distúrbio isolado.

Para identificar se ocorreu um agarramento em um distúrbio isolado em um trecho, observar os seguintes aspectos:

- Obrigatoriamente não deve existir alteração do SP em instantes próximos ao instante inicial ti do distúrbio. Caso exista alteração do SP em instantes próximos ao instante inicial ti do distúrbio,
não é um distúrbio isolado.
- Distúrbios isolados possuem um número de zero-crossings menor que 2.
- Tempo de reação da PV (TrPV) maior ou igual a (1.5/Tam) vezes o tempo de reação da MV (TrMV).


--- Exemplo 1 de comportamento específico de agarramento em um distúrbio isolado ---

Utilizar o critério C2%.

Sendo as Variáveis:

t0: SP = 20.0, PV = 20.0, MV = 35.5, t = 50.0s
t1: SP = 20.0, PV = 20.1, MV = 35.5, t = 50.5s
t2: SP = 20.0, PV = 20.2, MV = 35.5, t = 51.0s
t3: SP = 20.0, PV = 20.2, MV = 35.4, t = 51.5s
t4: SP = 20.0, PV = 20.2, MV = 35.5, t = 52.0s
t5: SP = 20.0, PV = 20.1, MV = 35.5, t = 52.5s
t6: SP = 20.0, PV = 20.1, MV = 35.5, t = 53.0s
t7: SP = 20.0, PV = 20.0, MV = 35.5, t = 53.5s
t8: SP = 20.0, PV = 19.9, MV = 35.5, t = 54.0s
t9: SP = 20.0, PV = 19.8, MV = 35.6, t = 54.5s
t10: SP = 20.0, PV = 19.7, MV = 35.6, t = 55.0s
t11: SP = 20.0, PV = 19.9, MV = 35.7, t = 55.5s
t12: SP = 20.0, PV = 14.8, MV = 35.5, t = 56.0s
t13: SP = 20.0, PV = 14.8, MV = 48.5, t = 56.5s
t14: SP = 20.0, PV = 14.9, MV = 48.5, t = 57.0s
t15: SP = 20.0, PV = 14.9, MV = 48.5, t = 57.5s
t16: SP = 20.0, PV = 14.8, MV = 48.5, t = 58.0s
t17: SP = 20.0, PV = 14.8, MV = 48.5, t = 58.5s
t18: SP = 20.0, PV = 14.8, MV = 61.3, t = 59.0s
t19: SP = 20.0, PV = 14.9, MV = 61.3, t = 59.5s
t20: SP = 20.0, PV = 14.9, MV = 61.3, t = 60.0s
t21: SP = 20.0, PV = 14.8, MV = 61.3, t = 60.5s
t22: SP = 20.0, PV = 14.8, MV = 61.3, t = 61.0s
t23: SP = 20.0, PV = 14.9, MV = 74.2, t = 61.5s
t24: SP = 20.0, PV = 14.9, MV = 74.2, t = 62.0s
t25: SP = 20.0, PV = 14.9, MV = 74.2, t = 62.5s
t26: SP = 20.0, PV = 14.9, MV = 74.2, t = 63.0s
t27: SP = 20.0, PV = 14.9, MV = 87.3, t = 63.5s
t28: SP = 20.0, PV = 15.0, MV = 87.3, t = 64.0s
t29: SP = 20.0, PV = 15.0, MV = 87.3, t = 64.5s
t30: SP = 20.0, PV = 15.0, MV = 87.3, t = 65.0s
t31: SP = 20.0, PV = 15.0, MV = 100.0, t = 65.5s
t32: SP = 20.0, PV = 14.9, MV = 100.0, t = 66.0s
t33: SP = 20.0, PV = 14.9, MV = 100.0, t = 66.5s
t34: SP = 20.0, PV = 14.9, MV = 100.0, t = 67.0s
t35: SP = 20.0, PV = 17.1, MV = 100.0, t = 67.5s
t36: SP = 20.0, PV = 19.8, MV = 100.0, t = 68.0s
t37: SP = 20.0, PV = 26.2, MV = 100.0, t = 68.5s
t38: SP = 20.0, PV = 33.7, MV = 100.0, t = 69.0s
t39: SP = 20.0, PV = 35.9, MV = 65.7, t = 69.5s
t40: SP = 20.0, PV = 26.2, MV = 65.7, t = 70.0s
t41: SP = 20.0, PV = 24.1, MV = 65.7, t = 70.5s
t42: SP = 20.0, PV = 22.4, MV = 65.7, t = 71.0s
t43: SP = 20.0, PV = 21.5, MV = 59.9, t = 71.5s
t44: SP = 20.0, PV = 20.6, MV = 59.9, t = 72.0s
t45: SP = 20.0, PV = 20.3, MV = 59.9, t = 72.5s
t46: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.0s
t47: SP = 20.0, PV = 20.2, MV = 59.9, t = 73.5s
t48: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.0s
t49: SP = 20.0, PV = 20.2, MV = 59.9, t = 74.5s
t50: SP = 20.0, PV = 20.2, MV = 59.9, t = 75.0s
t51: SP = 20.0, PV = 20.2, MV = 59.9, t = 75.5s
t52: SP = 20.0, PV = 20.1, MV = 59.9, t = 76.0s
t53: SP = 20.0, PV = 20.1, MV = 59.9, t = 76.5s
t54: SP = 20.0, PV = 20.1, MV = 59.9, t = 77.0s
t55: SP = 20.0, PV = 20.1, MV = 59.9, t = 77.5s
t56: SP = 20.0, PV = 20.1, MV = 59.9, t = 78.0s
t57: SP = 20.0, PV = 20.1, MV = 59.9, t = 78.5s
t58: SP = 20.0, PV = 20.0, MV = 59.9, t = 79.0s
t59: SP = 20.0, PV = 20.0, MV = 59.9, t = 79.5s
t60: SP = 20.0, PV = 20.0, MV = 59.9, t = 80.0s
t61: SP = 20.0, PV = 20.0, MV = 59.9, t = 80.5s
t62: SP = 20.0, PV = 20.0, MV = 59.9, t = 81.0s
t63: SP = 20.0, PV = 20.0, MV = 59.9, t = 81.5s
t64: SP = 20.0, PV = 20.0, MV = 59.9, t = 82.0s
t65: SP = 20.0, PV = 20.0, MV = 59.9, t = 82.5s
t66: SP = 20.0, PV = 20.0, MV = 59.9, t = 83.0s

Então, calcula-se o módulo do erro (|erro|) e compara-se o valor de |erro| com o valor de
2% do SP para cada instante de tempo. Caso o valor do |erro| seja maior que 2% do SP, então avalia-se também
o tempo de reação da MV e o tempo de reação da PV para verificar se há indícios de agarramento durante o intervalo ti e tf do distúrbio.
De acordo com as respectivas variáveis, tem-se:

t0: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t1: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t2: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t3: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t4: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t5: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t6: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t7: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.00)>(0.4) = False
t8: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t9: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t10: |erro| = |SP - PV| = |20.0 - 19.7| = | 0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t11: |erro| = |SP - PV| = |20.0 - 19.9| = | 0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t12: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t13: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t14: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t15: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t16: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t17: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t18: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t19: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t20: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t21: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t22: |erro| = |SP - PV| = |20.0 - 14.8| = | 5.20| = 5.20, (|erro|)>(2% do SP) = (5.20)>(0.4) = True
t23: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t24: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t25: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t26: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t27: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t28: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t29: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t30: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t31: |erro| = |SP - PV| = |20.0 - 15.0| = | 5.00| = 5.00, (|erro|)>(2% do SP) = (5.00)>(0.4) = True
t32: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t33: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t34: |erro| = |SP - PV| = |20.0 - 14.9| = | 5.10| = 5.10, (|erro|)>(2% do SP) = (5.10)>(0.4) = True
t35: |erro| = |SP - PV| = |20.0 - 17.1| = | 2.90| = 2.90, (|erro|)>(2% do SP) = (2.90)>(0.4) = True
t36: |erro| = |SP - PV| = |20.0 - 19.8| = | 0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t37: |erro| = |SP - PV| = |20.0 - 26.2| = |-6.20| = 6.20, (|erro|)>(2% do SP) = (6.20)>(0.4) = True
t38: |erro| = |SP - PV| = |20.0 - 33.7| = |-13.70| = 13.70, (|erro|)>(2% do SP) = (13.70)>(0.4) = True
t39: |erro| = |SP - PV| = |20.0 - 35.9| = |-15.90| = 15.90, (|erro|)>(2% do SP) = (15.90)>(0.4) = True
t40: |erro| = |SP - PV| = |20.0 - 26.2| = |-6.20| = 6.20, (|erro|)>(2% do SP) = (6.20)>(0.4) = True
t41: |erro| = |SP - PV| = |20.0 - 24.1| = |-4.10| = 4.10, (|erro|)>(2% do SP) = (4.10)>(0.4) = True
t42: |erro| = |SP - PV| = |20.0 - 22.4| = |-2.40| = 2.40, (|erro|)>(2% do SP) = (2.40)>(0.4) = True
t43: |erro| = |SP - PV| = |20.0 - 21.5| = |-1.50| = 1.50, (|erro|)>(2% do SP) = (1.50)>(0.4) = True
t44: |erro| = |SP - PV| = |20.0 - 20.6| = |-0.60| = 0.60, (|erro|)>(2% do SP) = (0.60)>(0.4) = True
t45: |erro| = |SP - PV| = |20.0 - 20.3| = |-0.30| = 0.30, (|erro|)>(2% do SP) = (0.30)>(0.4) = False
t46: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t47: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t48: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t49: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t50: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t51: |erro| = |SP - PV| = |20.0 - 20.2| = |-0.20| = 0.20, (|erro|)>(2% do SP) = (0.20)>(0.4) = False
t52: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t53: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t54: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t55: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t56: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t57: |erro| = |SP - PV| = |20.0 - 20.1| = |-0.10| = 0.10, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t58: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t59: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t60: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t61: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t62: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t63: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t64: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t65: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False
t66: |erro| = |SP - PV| = |20.0 - 20.0| = | 0.00| = 0.00, (|erro|)>(2% do SP) = (0.10)>(0.4) = False

O tempo de amostragem calculado é de Tam = 0.5s e o número de N de instantes de tempo é N = 20 instantes de tempo.

Foi identificado 01 distúrbio.

O instante inicial do distúrbio 01 é em ti1=56.0s, pois é o primeiro instante de tempo em que o |erro| foi maior que 2% do valor do SP.
O instante final do distúrbio 01 é em tf1=72.5s, pois é o primeiro instante de tempo após ti em que o |erro| foi menor que 2% do valor do SP E
os N = 20 instantes de tempo seguintes possuem valores de |erro| menores do que 2%.

O distúrbio 01 ocorre do instante t=56.0s a t=72.5s.

O número de Zero-crossings para o distúrbio identificado foi de 01 zero-crossings, no instante 68.0s. 

Com o distúrbio devidamente identificado, realiza-se a classificação do distúrbio:

O distúrbio 01 ocorre do instante ti1=56.0s a tf1=72.5s, é um distúrbio isolado pois ocorre um valor de erro entre o SP e a PV sem a alteração do SP, seguido
de uma ação na MV (instante t=56.5s) para que seja ajustada a PV, e após instante t=72.5s (tf1=72.5s) ocorrem N = 20 instantes de tempo com valores de |erro| menores do que o critério C2%, com
ocorrência de 01 zero-crossings, no instante 68.0s.

Então, o distúrbio 01 é classificado como Distúrbio Isolado.

Com o distúrbio devidamente identificado e classificado, verifica-se se há evidências de agarramento:

Dentro do intervalo t=56.0s a t=72.5s do distúrbio 01 identificado, o instante da primeira reação da MV foi em t=56.5s, portanto o TrMV=0.5s, pois a MV reagiu 0.5 segundos
após o início do distúrbio. ti1=56.0s. O instante da primeira reação da PV foi em t=67.5s, portanto o TrPV=11.5s, pois a PV reagiu 11.5 segundos após o início do distúrbio ti1=56.0s.

Então:

TrPV = 11.5s
TrMV = 0.5s

(TrPV) <= ((1.5/Tam)*TrMV) = (11.5)<=((1.5/0.5)*0.5) = (11.5)<=(1.5) = False. Então, há indícios de agarramento.

A Conclusão é que houve 01 trecho de distúrbio identificado do instante t=54.0s a t=71.0s, classificado como Distúrbio Isolado e há evidência de agarramento.

--- Fim do Exemplo 1 de comportamento específico de agarramento em um distúrbio isolado ---


--- Oscilações causadas por agarramento ---

Agarramento ou Stiction prejudica o movimento adequado da válvula, onde a haste da válvula não consegue se mover em
resposta ao sinal de saída do controlador (MV), o que causa a necessidade de um crescimento gradativo do sinal da MV
para conseguir que a haste se mova, e após um determinado período de tempo, ocorre um movimento abrupto de desprendimento
da haste da válvula para resposta ao sinal do controlador (MV). Este salto é denominado slip-jump.

Quando o grau de intensidade do agarramento está muito elevado (ou fora do nível adequado), são observadas oscilações
autoalimentadas e permanentes na malha de controle, denominadas Limit Cycles, que são um comportamento oscilatório estável,
onde não são amortecidos, não aumentam indefinidamente e se repetem continuamente. Estas oscilações Limit cycles são causadas
exclusivamente pelo efeito de agarramento da válvula, e não pelo processo (planta) ou pelo controlador. As reações Limit Cycle
podem ser observadas mesmo com o Set Point (SP) fixo.

O agarramento quando em grau elevado na válvula de controle, provoca limit cycles, que são ciclos repetitivos de erro e correção
e geram o seguinte impacto na malha de controle:

1 – Acumula o erro (devido à fase de deadband da válvula de controle)
2 – Aumenta a MV até vencer o atrito estático.
3 – Ocorrência do efeito de slip-jump, causando um salto na abertura da válvula de controle.
4 – Gera um novo erro.
5 – Repete o ciclo.

Este impacto descrito na malha de controle é o que gera as oscilações periódicas e estáveis, denominadas como limit cycles, que
podem ser formas de onda quadradas (square), triangulares, dente de serra (saw-tooth), senoidal. Estas formas de onda não possuem
o formato perfeito, porém se aparentam com as formas citadas. Estas formas de onda quando induzidas por agarramento na válvula de
controle, contém alto conteúdo harmônico em seus sinais de MV e PV. 

Quando as oscilações na MV e na PV são causadas por outros motivadores que não agarramento, como sintonia deficiente do controlador
ou distúrbio externo, MV e PV possuem forma de onda senoidal com baixo conteúdo harmônico.

O fenômeno de agarramento/stiction gera limit cycles que deixa marcas distintas na forma de onda da PV e da MV, podendo ser
caracterizadas de acordo com os seguintes padrões típicos:

- Forma de onda Triangular para MV e Quadrada para PV;
- Forma de onda Retangular para MV e Retangular para PV;
- Forma de onda Triangular para MV e Senoidal para PV;
- Forma de onda Triangular para MV e Triangular para PV;

Para identificação de agarramento, estes padrões típicos para limit cycles apresentados devem obrigatoriamente serem respeitados,
caso não sejam levados em consideração, há grande chance de imprecisão na avaliação. São estes padrões típicos de limit cycle que
diferenciam o agarramento de outros motivos de oscilações, como distúrbios externos ou má sintonia do controlador.

<==== FIM DAS TÉCNICAS PARA AVALIAÇÃO DE AGARRAMENTO EM VÁLVULAS DE CONTROLE


====> INÍCIO DA ESPECIFICAÇÃO DAS PREMISSAS PARA ELABORAÇÃO DO RELATÓRIO TÉCNICO DA ANÁLISE

O Relatório Técnico da Análise, deverá seguir as seguintes premissas, sem exceção:

- Deve conter o Título.
- Deve conter o Cabeçalho.
- Deve conter a Figura do Gráfico do Resultado.
- Deve conter o Texto técnico do diagnóstico da Avaliação. 

--- Especificação do Título do Relatório ---

O Título deve ser “Relatório Técnico de Análise de Malha de Controle” e deve estar centralizado.


--- Especificação do Cabeçalho do Relatório ---

O Cabeçalho do relatório deve conter os seguintes dados:

- Abaixo do título, deve estar escrito o TAG da malha de controle.
- Abaixo do TAG deve estar o nome do Engenheiro Responsável: “Márlon A. B. Damasceno”
- Abaixo do nome do Engenheiro Responsável, deve estar o nome da instituição: “UFOP – Universidade Federal de Ouro Preto”
- Abaixo do nome da instituição, inserir a data do relatório.
- O cabeçalho deve estar alinhado à esquerda.


--- Especificação da Figura do Gráfico do Resultado ---

A Figura do Gráfico do Resultado deve ser inserida no relatório abaixo do cabeçalho, da seguinte forma:

- A Figura deve ser 01 imagem somente e deve ser o gráfico do resultado.
- A Figura deve estar centralizada.
- A Legenda da descrição da Figura deve ser “Gráfico de Amostra de Sinal – Avaliação de agarramentos em Malha de Controle”.
- A Legenda da descrição da Figura deve ser posicionada abaixo da Figura e centralizada.
- A Legenda da descrição da Figura deve ser escrita no padrão ABNT.


--- Especificação do Gráfico do Resultado ---

O Gráfico do Resultado deve atender as seguintes premissas, sem exceção:

- O Gráfico do Resultado deverá representar o resultado da amostra completa.
- O Gráfico deve ser apenas 01 (uma) Figura que contenha todos os sinais: PV, MV e SP. Os sinais de MV, SP, e PV não devem ser apresentados em gráficos separados.
- Os sinais de MV, PV e SP devem ser apresentados simultaneamente na mesma figura de gráfico.
- O sinal de MV deve ser representado por uma linha contínua na cor Azul, código hexadecimal #1F4FD8.
- O sinal de PV deve ser representado por uma linha contínua na cor Vermelha, código hexadecimal #E74C3C.
- O sinal de SP deve ser representado por uma linha tracejada na cor Verde, código hexadecimal #2ECC71.
- O fundo dos trechos onde há distúrbios com evidência de agarramento DEVE ser preenchido com a cor Amarela, código hexadecimal #F1C40F, para destacar e permitir
melhor visualização e entendimento do gráfico pelo usuário.
- Devem ser inseridos marcadores para delimitar o início do(s) distúrbio(s) com evidência de agarramento e o final do(s) distúrbio(s) com evidência de agarrament, que deverão ser
representados por linhas verticais tracejadas na cor Cinza, código hexadecimal #2C2C2C, com marcador/label de “Início” para o delimitador do início do distúrbio com evidência de agarramento
e marcador/label de “Término” para o delimitador de final do distúrbio com evidência de agarramento para permitir melhor visualização e entendimento do gráfico pelo usuário.
- A Legenda do Gráfico deve ser apresentada com a identificação dos sinais de SP, MV, PV e Trecho com distúrbio(s) com evidência de agarramento (caso haja algum trecho com distúrbio com evidência de agarramento). 
- Os sinais de MV, SP, e PV não devem ser representados por outras cores.
- A imagem gerada para este Gráfico de Resultado deve ser a mesma imagem utilizada no Relatório Técnico da Análise.


--- Especificação do Texto técnico do diagnóstico da Avaliação ---

- O Texto técnico do diagnóstico da Avaliação deve vir abaixo da Figura do Gráfico do Resultado.
- O Texto técnico do diagnóstico da Avaliação deve ser resumido em 01 parágrafo de 15 linhas.
- O Texto técnico do diagnóstico da Avaliação deve conter apenas o resultado final da avaliação técnica. Não é necessário descrever o passo a passo ou
o raciocínio que foi feito para obter os resultados no texto.
- O Texto técnico do diagnóstico da Avaliação deve informar os instantes iniciais e finais de cada trecho de distúrbio com evidência de agarramento detectado na malha de controle.
- Ao final do Texto técnico do diagnóstico da Avaliação deve estar a conclusão final informando se foi(foram) constatado(constatados) distúrbio(s) com evidência de agarramento no sistema de controle ou
se não foi(foram) constatado(constatados) distúrbio(s) com evidência de agarramento no sistema de controle.

<==== FIM DA ESPECIFICAÇÃO DAS PREMISSAS PARA ELABORAÇÃO DO RELATÓRIO TÉCNICO DA ANÁLISE

Para realizar sua tarefa de identificar se há agarramento na válvula de controle ou se não há agarramento na válvula de controle da malha de controle da respectiva a amostra recebida pelo arquivo .csv, execute as
seguintes subtarefas:

1. **Identificar trechos com distúrbio:** Comece identificando em qual(is) trecho(s) houveram distúrbios e determine o instante de início e o instante de término para cada distúrbio identificado.

    a. **Identificar o instante ti inicial do trecho do distúrbio:** Considere que o instante inicial do trecho do distúrbio identificado é o instante de tempo t em que o valor do módulo do erro
    entre SP e PV (|erro| = SP - PV) seja pelo critério C2%. Lembre-se da ORIENTAÇÃO TÉCNICA PARA IDENTIFICAÇÃO DE DISTÚRBIOS.
    
    b. **Identificar o instante tf final do trecho do distúrbio:** Considere que o instante final do trecho do distúrbio identificado é quando ocorrer de o valor do módulo do erro
    entre SP e PV (|erro| = SP - PV) seja pelo critério C1% em tf, e também seja pelo critério C1% para N instantes de tempo seguintes, onde N depende do Tempo de Amostragem Tam da amostra fornecida.
    Lembre-se da ORIENTAÇÃO TÉCNICA PARA IDENTIFICAÇÃO DE DISTÚRBIOS.

2 - **Classificar os trechos com distúrbio:** Classifique quais trechos identificados como distúrbio são transientes, quais trechos são distúrbios isolados e quais trechos são oscilações.

    a. **Verificar se houve alteração do SP:** Verifique se no instante inicial do trecho do distúrbio identificado houve alteração no valor do SP. Lembre-se das TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS.
    
    b. **Avaliar o número de zero-crossings:** Verifique se o número de ocorrências de zero-crossings seja maior ou menor do que o número limite de ocorrências de zero-crossings determinado.
    Lembre-se das TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS.

    c. **Definir quais trechos com distúrbio são transientes:** Avalie quais trechos de distúrbio possuem características de transientes - alteração de SP e número de zero-crossings menor que 2.
    Lembre-se das TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS.

    d. **Definir quais trechos com distúrbio são distúrbios isolados:** Avalie quais trechos de distúrbio possuem características de distúrbios isolados - SP constante (sem alteração) e número de
    zero-crossings menor que 2. Lembre-se das TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS.

    e. **Definir quais trechos com distúrbio são oscilações:** Avalie quais trechos de distúrbio possuem características de oscilações - número de zero-crossings maior que 2.
    Lembre-se das TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS.

3 - **Avaliar quais distúrbios possuem evidências de agarramento:** Verifique quais os trechos com distúrbio possuem as características que evidenciem agarramento (sticion) na válvula de controle.
Lembre-se das TÉCNICAS PARA AVALIAÇÃO DE AGARRAMENTO EM VÁLVULAS DE CONTROLE.

    a. **Verificar agarramento nos distúrbios classificados como transientes:** Verifique se há indícios de agarramento nos trechos de distúrbio classificados como transientes.
    Lembre-se das TÉCNICAS PARA AVALIAÇÃO DE AGARRAMENTO EM VÁLVULAS DE CONTROLE.
    
    b. **Verificar agarramento nos distúrbios classificados como distúrbios isolados:** Verifique se há indícios de agarramento nos trechos de distúrbio classificados como distúrbios isolados.
    Lembre-se das TÉCNICAS PARA AVALIAÇÃO DE AGARRAMENTO EM VÁLVULAS DE CONTROLE.

    c. **Verificar agarramento nos distúrbios classificados como oscilações:** Verifique se há indícios de agarramento nos trechos de distúrbio classificados como oscilações.
    Lembre-se das TÉCNICAS PARA AVALIAÇÃO DE AGARRAMENTO EM VÁLVULAS DE CONTROLE.

4 - **Destacar os trechos de distúrbios com evidência de agarramento:** Destaque os trechos de distúrbio com evidência de agarramento. Cada trecho de distúrbio possui
um instante inicial ti e um instante final tf. Lembre-se das TÉCNICAS PARA AVALIAÇÃO DE AGARRAMENTO EM VÁLVULAS DE CONTROLE.

5 – **Gerar o Relatório Técnico:** Gere o Relatório Técnico da análise conforme modelo solicitado e apresente o(s) distúrbio(s) com agarramento(s) detectado(s) na amostra em destaque.
Lembre-se DA ESPECIFICAÇÃO DAS PREMISSAS PARA ELABORAÇÃO DO RELATÓRIO TÉCNICO DA ANÁLISE.

"""

#    c. **Avaliar a amplitude da oscilação:** Caso o número de ocorrências de zero-crossings seja maior do que o número limite de ocorrências de zero-crossings determinado, avalie se a amplitude
#    da oscilação é amortecida ou se não é amortecida. Lembre-se das TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS.

#    d. **Avaliar o período da oscilação:** Caso o número de ocorrências de zero-crossings seja maior do que o número limite de ocorrências de zero-crossings determinado, avalie se o período da
#    oscilação é considerado constante ou se não é considerado constante. Lembre-se das TÉCNICAS PARA CLASSIFICAÇÃO DE DISTÚRBIOS.

def wrap_text_to_width(text, font_name, font_size, max_width):
    """
    Quebra o texto em linhas que cabem em max_width (em pontos),
    medindo com a fonte informada, para não exceder as margens.
    """
    words = re.sub(r"\s+", " ", text.strip()).split(" ")
    lines = []
    current = ""

    for w in words:
        test = w if not current else f"{current} {w}"
        if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w

    if current:
        lines.append(current)

    return lines

def limpar_markdown_basico(s: str) -> str:
    """
    Limpa os markdowns que aparecem no texto do diagnóstico para que não apareçam no texto do relatório quando extraídos.
    """    
    # limpa os marcadores que sinalizam negrito/itálico com asteriscos e underscores
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)  # de **texto** para texto (um markdown para negrito)
    s = re.sub(r"\*(.*?)\*", r"\1", s)      # de *texto* para  texto (um markdown para negrito)
    s = re.sub(r"__(.*?)__", r"\1", s)      # de __texto__ para texto (um markdown para itálico)
    s = re.sub(r"_(.*?)_", r"\1", s)        # de _texto_ para texto (um markdown para itálico)

    # remove markdowns para cabeçalho (jogo da velha e >)
    s = re.sub(r"^\s*#+\s*", "", s, flags=re.MULTILINE)  # de # Título para Título
    s = re.sub(r"^\s*>\s*", "", s, flags=re.MULTILINE)   # de > citação para citação

    return s

def draw_justified_line(c, x_left, y, line, font_name, font_size, max_width):
    """
    Desenha uma linha justificada.
    """
    # Se não tem espaço pra distribuir, desenha normal
    if " " not in line:
        c.drawString(x_left, y, line)
        return

    words = line.split(" ")
    # largura das palavras (sem espaços)
    words_width = sum(pdfmetrics.stringWidth(w, font_name, font_size) for w in words)
    gaps = len(words) - 1
    if gaps <= 0:
        c.drawString(x_left, y, line)
        return

    extra = max_width - words_width
    if extra <= 0:
        c.drawString(x_left, y, line)
        return

    space = extra / gaps

    x = x_left
    for i, w in enumerate(words):
        c.drawString(x, y, w)
        x += pdfmetrics.stringWidth(w, font_name, font_size)
        if i < gaps:
            x += space

def extrair_tag(texto, fallback="TAG_NAO_IDENTIFICADA"):
    # filtra o TAG do arquivo de amostra
    m = re.search(r"\bTAG\b\s*[:\-]\s*([A-Za-z0-9_.\-\/]+)", texto)
    if m:
        return m.group(1).strip()
    m = re.search(r"\*\*TAG:\*\*\s*([A-Za-z0-9_.\-\/]+)", texto)
    if m:
        return m.group(1).strip()
    return fallback

def extrair_paragrafo_avaliacao(texto):
    """
    Extrai o parágrafo principal de avaliação gerado pela LLM.
    """
    m = re.search(r"(Avaliação técnica:.*)", texto, flags=re.IGNORECASE | re.DOTALL)
    if m:
        bloco = m.group(1).strip()
        bloco = bloco.split("\n\n")[0].strip()
        return bloco

    # Lê os markdowns e extrai o último parágrafo
    partes = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    if partes:
        return max(partes, key=len)

    return texto.strip()

def gerar_pdf_local(pdf_path, tag, data_str, imagem_path, paragrafo):
    """
    Monta o PDF local (usando o ReportLab) com os seguintes requisitos:
    - Título: centralizado
    
    Cabeçalho (similar ao solicitado pelo prompt):
    
    - TAG da Malha de Controle.
    - Data do Relatório.
    - Nome do Engenheiro Responsável.
    - Instituição.
    - Gráfico (Imagem): centralizado, com legenda padrão ABNT e abaixo da Figura.
    - Texto descritivo em 01 parágrafo formatado ABNT.

    O texto, os dados do cabeçalho, e todo conteúdo do relatório é somente montado aqui, não sendo
    gerado nenhum conteudo textual ou imagem, apenas inserido para o relatório. Quem gera o texto e
    a imagem é a LLM
    
    """
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Configuração das Margens
    margem_esq = 3*cm
    margem_dir = 2*cm
    margem_sup = 3*cm
    margem_inf = 2*cm

    y = h - margem_sup

    # Título
    c.setFont("Times-Bold", 14)
    c.drawCentredString(w/2, y, "Relatório Técnico de Análise de Malha de Controle")
    y -= 1.0*cm

    # Cabeçalho alinhado à esquerda
    c.setFont("Times-Roman", 12)
    x_esq = margem_esq  # alinhamento pela margem esquerda
    c.drawString(x_esq, y, f"TAG: {tag}    |    Data: {data_str}")
    y -= 0.6 * cm
    c.drawString(x_esq, y, "Engenheiro Responsável: Márlon A. B. Damasceno")
    y -= 0.6 * cm
    c.drawString(x_esq, y, "UFOP – Universidade Federal de Ouro Preto")
    y -= 1.0 * cm

    # Imagem
    img = ImageReader(imagem_path)
    iw, ih = img.getSize()

    max_w = w - (margem_esq + margem_dir)
    max_h = 10.5*cm  # altura máxima para a figura

    scale = min(max_w/iw, max_h/ih)
    dw, dh = iw*scale, ih*scale

    x_img = (w - dw)/2
    y_img = y - dh

    c.drawImage(img, x_img, y_img, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
    y = y_img - 0.6*cm

    # Legenda
    c.setFont("Times-Italic", 12)
    c.drawCentredString(
        w/2,
        y,
        "Figura 1 – Gráfico de Amostra de Sinal – Avaliação de Agarramento/Stiction em Válvula de Controle."
    )
    y -= 1.0*cm

    # Parágrafo 
    fonte = "Times-Roman"
    tamanho_fonte = 12
    espacamento = 14  # espaçamento entre linhas

    c.setFont(fonte, tamanho_fonte)
    largura_util = w - (margem_esq + margem_dir)
    paragrafo = limpar_markdown_basico(paragrafo)
    lines = wrap_text_to_width(paragrafo, fonte, tamanho_fonte, largura_util)

    y_text = y

    for i, line in enumerate(lines):
        # nova página se necessário (mas o padrão é relatório de 01 página)
        if y_text < margem_inf + 1.5*cm:
            c.showPage()
            y_text = h - margem_sup
            c.setFont(fonte, tamanho_fonte)

        # Justifica linhas
        if i < len(lines) - 1:
            draw_justified_line(c, margem_esq, y_text, line, fonte, tamanho_fonte, largura_util)
        else:
            c.drawString(margem_esq, y_text, line)

        y_text -= espacamento

    c.showPage()
    c.save()

def listar_arquivos_com_annotations(resp_dict):
    imagens = []
    pdfs = []
    outros = []

    # Saídas diretas do Code Interpreter - mesmo indicando esta etapa, a LLM que decide se usa ou não o Code Interpreter
    for item in resp_dict.get("output", []):
        if item.get("type") == "code_interpreter_call":
            for out in item.get("outputs", []):
                if not isinstance(out, dict):
                    continue
                out_type = out.get("type")
                url = out.get("url", "")
                filename = (out.get("filename") or out.get("name") or "").lower()
                mime = (out.get("mime_type") or out.get("content_type") or "").lower()

                if out_type == "image" and url:
                    imagens.append(out)
                elif out_type == "file":
                    if ("pdf" in mime) or filename.endswith(".pdf") or url.lower().endswith(".pdf"):
                        pdfs.append(out)
                    else:
                        outros.append(out)

    # Filtra PDF's anexados na saída da LLM
    for item in resp_dict.get("output", []):
        if item.get("type") != "message":
            continue

        for c in item.get("content", []):
            if not isinstance(c, dict):
                continue
            if c.get("type") != "output_text":
                continue

            text = c.get("text", "")
            annotations = c.get("annotations", []) or []
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                if ann.get("type") in ("file_path", "file"):
                    ann_text = (ann.get("text") or "").lower()
                    file_id = ann.get("file_id") or ann.get("id")
                    if file_id and ann_text.endswith(".pdf"):
                        pdfs.append({
                            "type": "file",
                            "file_id": file_id,
                            "filename": os.path.basename(ann.get("text", "relatorio.pdf")),
                            "source": "annotation",
                            "raw": ann,
                            "text_snippet": text[:200],
                        })

    return {"imagens": imagens, "pdfs": pdfs, "outros": outros}


def listar_arquivos(resp_dict):
    """
    Lista somente arquivos gerados pelo Code Interpreter.
    Retorna um dicionário com imagens e PDFs separados.

    Estrutura de retorno:
    {
        "imagens": [ {dict do output} ],
        "pdfs":    [ {dict do output} ],
        "outros":  [ {dict do output} ]  # opcional
    }
    """
    imagens = []
    pdfs = []
    outros = []

    for item in resp_dict.get("output", []):
        if item.get("type") != "code_interpreter_call":
            continue

        for out in item.get("outputs", []):
            if not isinstance(out, dict):
                continue

            out_type = out.get("type")
            url = out.get("url", "")
            filename = (out.get("filename") or out.get("name") or "").lower()
            mime = (out.get("mime_type") or out.get("content_type") or "").lower()

            # Lista de imagens da saída - imagens que a LLM gerou e anexou como saída
            if out_type == "image" and url:
                imagens.append(out)

            # Lista de PDFs da saída - arquivos PDF que a LLM gerou e anexou como saída (não está anexando por algum motivo)
            elif (
                out_type == "file"
                and (
                    "pdf" in mime
                    or filename.endswith(".pdf")
                    or url.lower().endswith(".pdf")
                )
            ):
                pdfs.append(out)

            # Outras extensões de arquivos que por ventura a LLM venha a anexar
            elif out_type == "file":
                outros.append(out)

    return {
        "imagens": imagens,
        "pdfs": pdfs,
        "outros": outros,
    }



def extrair_imagem_arquivo(arquivos_gerados, nome="grafico_stiction.png", timeout=30):
    imagens = arquivos_gerados.get("imagens", [])
    if not imagens:
        return None

    img_out = imagens[0]  # primeira imagem
    tmpdir = tempfile.mkdtemp()
    img_path = os.path.join(tmpdir, nome)

    # URL do container do servidor da respectiva LLM (Exemplo: "data:image/png;base64", "sandbox", etc.)
    url = img_out.get("url") if isinstance(img_out, dict) else None
    if isinstance(url, str) and url.startswith("data:image/") and "base64," in url:
        b64 = url.split("base64,", 1)[1]
        b64 = re.sub(r"\s+", "", b64)
        img_bytes = base64.b64decode(b64)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        return img_path

    # URL de webpage (Exemplo: "http", "https", etc...)
    if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(r.content)
        return img_path

    # codificado em base64 (Exemplo: "data", "image", etc.)
    b64 = None
    if isinstance(img_out, dict):
        if isinstance(img_out.get("data"), str):
            b64 = img_out["data"]
        elif isinstance(img_out.get("image"), dict) and isinstance(img_out["image"].get("data"), str):
            b64 = img_out["image"]["data"]

    if b64:
        b64 = re.sub(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", "", b64.strip())
        b64 = re.sub(r"\s+", "", b64)
        img_bytes = base64.b64decode(b64)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        return img_path

    # Caso não tenha conseguido extrair, emite a mensagem de erro
    raise ValueError(f"Imagem encontrada, mas formato não suportado. Chaves: {list(img_out.keys())}, url: {str(url)[:60]}...")


def run_analysis(uploaded_file, ciclos_analises):

    if uploaded_file is None:
        return "Nenhum arquivo enviado.", "Envie um arquivo .csv.", None

    csv_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name    

    # Cria ap asta local para armazenamento do(s) relatório(s) de testagem
    base_dir = os.path.join(os.getcwd(), "resultados")
    os.makedirs(base_dir, exist_ok=True)

    pdfs_gerados = []
    
    for i in range(1, ciclos_analises + 1):
        prompt = PROMPT_FIXO

        # carrega o CSV para a Files API - para não carregar o arquivo na janela de contexto
        file_obj = client.files.create(
            file=open(csv_path, "rb"),
            purpose="user_data",
        )

        # chamada da LLM
        resp = client.responses.create(
            model=OPENAI_MODEL,
            # Necessário incluir a ferramenta code interpreter, para que a LLM decida se executa os cálculos usando o code interpreter ou não
            # Como melhoria, adicionar um log para indicar que a LLM optou por usar a ferramenta (cao o usuário faça uma pergunta simples que não
            # demande o uso da ferramenta, evidenciando a decisão da LLM)
            tools=[{
                "type": "code_interpreter",
                "container": {"type": "auto", "memory_limit": "4g", "file_ids": [file_obj.id]}
            }],
            # o argumento "tool_choice="auto"" deixa à cargo da LLM a decisão de usar ou não a ferramenta. Já o argumento "tool_choice="required""
            # obriga (força) a LLM a sempre utilizar a ferramenta.
            tool_choice="auto",
            include=["code_interpreter_call.outputs"],
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }],
        )

        # Aquisição do texto de saída gerado pela LLM (somente o conteúdo textual, ou seja, o diagnóstico mesmo)
        text_out = getattr(resp, "output_text", "") or ""

        # Converte a resposta da LLM de texto estruturado JSON para um dicionário chave-valor (procedimento denominado "parse") 
        resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(json.dumps(resp))
        arquivos_gerados = listar_arquivos_com_annotations(resp_dict)

        # extrair a imagem do dicionário de resposta (executa o parse para que seja simplificada essa extração)
        imagem_out = extrair_imagem_arquivo(arquivos_gerados, nome="grafico_resultado_stiction.png")

        # Gera o relatório localmente com os tokens gerados pela LLM (apenas monta o relatório com o texto já gerado pela LLM)
        tag = extrair_tag(text_out, fallback="MALHA")
        data_str = datetime.now().strftime("%d/%m/%Y")
        paragrafo = extrair_paragrafo_avaliacao(text_out)

        # arquivo gerado por ciclo
        pdf_local_path = os.path.join(
            base_dir,
            f"{i:02d}_Relatorio_Tecnico_Analise_Malha_{tag}.pdf"
        )

        gerar_pdf_local(
            pdf_path=pdf_local_path,
            tag=tag,
            data_str=data_str,
            imagem_path=imagem_out,
            paragrafo=paragrafo,
        )

        print("Imagens encontradas:", len(arquivos_gerados["imagens"]))
        print("PDFs encontrados (API):", len(arquivos_gerados["pdfs"]))

        pdfs_gerados.append(pdf_local_path)

    return text_out, imagem_out, pdfs_gerados

    
###############################################################
## INTERFACE GRADIO (Blocks)
###############################################################

# Configura o modo Escuro (modo "Dark") na interface do Gradio
interface_escura = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

css = """
#pdf_output {
  min-height: 60px !important;
  height: 60px !important;
}


"""

with gr.Blocks(css=css, js=interface_escura) as demo:

    # Título e Cabeçalho da Interface
    gr.Markdown(f"""
<div>
  <h1 style="text-align: center;">Análise de Agarramento em Válvulas de Controle</h1>

  <p style="text-align: left;">
    Autor: Márlon A. B. Damasceno<br>
    UFOP – Universidade Federal de Ouro Preto<br>
    ITV – Instituto Tecnológico Vale<br><br>
    Modelo LLM: <b>{OPENAI_MODEL.upper()}</b>
  </p>
</div>
""")

    # Configuração da interface do Gradio - Blocos de Inputs e Outputs
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                in_csv = gr.File(label="Carregar a Amostra: Arquivo .csv", file_types=[".csv"], height=150, scale=2)
            ciclos_analises = gr.Slider(
                label="Número de Ciclos de Análise:",
                minimum=1,
                maximum=50,
                step=1,
                value=2,
                )

            botao_upload = gr.Button("Iniciar Análise da Amostra", variant="primary")  
            out_pdf = gr.File(label="Relatório Técnico de Análise de Malha de Controle", elem_id="pdf_output")
            
        with gr.Column(scale=2):
            out_img = gr.Image(label="Gráfico do Resultado da Análise da Amostra", type="filepath", height=285)

    with gr.Row():
        out_text = gr.Textbox(lines=18, label="Análise Técnica da Amostra") 

    # Ação do botão upload da interface Gradio
    botao_upload.click(
        fn=run_analysis,
        inputs=[in_csv, ciclos_analises],
        outputs=[out_text, out_img, out_pdf],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)

