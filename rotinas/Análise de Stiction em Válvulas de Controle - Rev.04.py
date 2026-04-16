#*****************
# 
# Rev.01: Versão Inicial
# Rev.02: Separado prompt fixo em partes e execução pausada
# Rev.03: Inclusão de delimitadores no prompt fixo
# Rev.04: Inclusão da repetição das instruções ao final, para tratar o viés de recência
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

Seu objetivo final é identificar se há ocorrência de agarramento por meio dos sinais de PV e MV fornecidos por um arquivo em extensão .csv
de amostra coletado de uma malha de controle e gerar um relatório técnico final.

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

====> INÍCIO DA ORIENTAÇÃO TÉCNICA PARA DIFERENCIAR OSCILAÇÕES E TRANSIENTES

--- Classificação das perturbações ---

A) Transientes e respostas a distúrbios externos
B) Oscilações autoalimentadas (limit cycles)

--- Critérios para classificar um trecho como OSCILAÇÃO (limit cycle) ---

- O comportamento deve ser aproximadamente periódico, com repetição clara de ciclos. Não há momentos de estabilidade em um trecho com oscilação.
- A amplitude das oscilações deve ser aproximadamente constante ao longo do tempo e não é amortecida.
- A oscilação deve persistir por vários períodos consecutivos. Considere como oscilação somente trechos cuja duração seja maior que, no mínimo, dois períodos
completos do comportamento observado. Trechos com menos de dois ciclos completos não devem ser classificados como oscilação, devem ser classificados como eventos isolados.
- De um modo geral, o Set Point (SP) deve estar constante ou quase constante durante o trecho analisado.
- Deve existir coerência temporal entre MV e PV compatível com comportamento não linear, como por exemplo rampas em MV e saltos na PV.
- Oscilações isoladas, amortecidas ou respostas a perturbações externas NÃO devem ser classificadas como limit cycles, e DEVEM ser desconsideradas pois
não são causadas por agarramento/stiction.
- Eventos transitórios, mesmo que apresentem mais de uma inversão de sinal, NÃO caracterizam oscilação se não houver repetição periódica e DEVEM ser desconsideradas pois
não são causadas por agarramento/stiction.

Somente trechos que atendam simultaneamente a TODOS os critérios acima podem ser considerados como oscilações candidatas a agarramento/stiction.

Os trechos que não atendem simultaneamente a TODOS os critérios acima não são oscilações.

<==== FIM DA ORIENTAÇÃO TÉCNICA PARA DIFERENCIAR OSCILAÇÕES E TRANSIENTES


====> INÍCIO DA ORIENTAÇÃO TÉCNICA PARA AVALIAR OSCILAÇÕES PELO EFEITO DE AGARRAMENTO

Para avaliar se a oscilação na forma de onda da PV e da MV são pelo efeito de agarramento, considerar os seguintes aspectos:

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

<==== FIM DA ORIENTAÇÃO TÉCNICA PARA AVALIAR OSCILAÇÕES PELO EFEITO DE AGARRAMENTO


====> INÍCIO DA ORIENTAÇÃO TÉCNICA PARA APLICAÇÃO DO MÉTODO DA CORRELAÇÃO CRUZADA

O Método da Correlação Cruzada permite avaliar se as oscilações da PV e da MV são causadas por agarramento na válvula de controle ou não.
O método consiste em aplicar a Função de Correlação Cruzada (CCF) entre o trecho em oscilação do sinal de MV e do sinal da PV, para
identificar se tais oscilações sinal foram causadas por agarramento da válvula de controle. Caso o resultado da Função de Correlação
Cruzada ao trecho aplicado seja uma função ímpar (assimétrica em relação ao eixo vertical), a causa provável da oscilação é agarramento
da válvula de controle. Caso o resultado da Função de Correlação Cruzada ao trecho aplicado seja uma função par (simétrica em relação ao
eixo vertical), é improvável que a oscilação foi causada por agarramento.

====> FIM DA ORIENTAÇÃO TÉCNICA PARA APLICAÇÃO DO MÉTODO DA CORRELAÇÃO CRUZADA


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
- A Legenda da descrição da Figura deve ser “Gráfico de Amostra de Sinal – Avaliação de Agarramento/Stiction em Válvula de Controle”.
- A Legenda da descrição da Figura deve ser posicionada abaixo da Figura e centralizada.
- A Legenda da descrição da Figura deve ser escrita no padrão ABNT.


--- Especificação do Gráfico do Resultado ---

O Gráfico do Resultado deve atender as seguintes premissas, sem exceção:

- O Gráfico do Resultado deverá representar o resultado da amostra completa.
- O Gráfico deverá ser apenas 01 (uma) Figura que contenha todos os sinais: PV, MV e SP. Os sinais de MV, SP, e PV não devem ser apresentados em gráficos separados.
- Os sinais de MV, PV e SP DEVEM ser apresentados simultaneamente na mesma figura de gráfico.
- O sinal de MV deve ser representado por uma linha contínua na cor Azul, código hexadecimal #1F4FD8.
- O sinal de PV deve ser representado por uma linha contínua na cor Vermelha, código hexadecimal #E74C3C.
- O sinal de SP deve ser representado por uma linha tracejada na cor Verde, código hexadecimal #2ECC71.
- O fundo dos trechos onde há agarramento DEVE ser preenchido com a cor Amarela, código hexadecimal #F1C40F, para destacar e permitir
melhor visualização e entendimento do gráfico pelo usuário.
- Devem ser inseridos marcadores para delimitar o início do agarramento e o final do agarramento, que deverão ser representados por linhas verticais tracejadas na cor
Cinza, código hexadecimal #2C2C2C, com marcador/label de “Início” para o delimitador do início do agarramento e marcador/label de “Término” para o delimitador
de final do agarramento para permitir melhor visualização e entendimento do gráfico pelo usuário.
- A Legenda do Gráfico deve ser apresentada com a identificação dos sinais de SP, MV, PV e Trecho com Agarramento (caso haja algum trecho com agarramento). 
- Os sinais de MV, SP, e PV não devem ser representados por outras cores.
- A imagem gerada para este Gráfico de Resultado deve ser a mesma imagem utilizada no Relatório Técnico da Análise.


--- Especificação do Texto técnico do diagnóstico da Avaliação ---

- O Texto técnico do diagnóstico da Avaliação deve vir abaixo da Figura do Gráfico do Resultado.
- O Texto técnico do diagnóstico da Avaliação deve ser resumido em 01 parágrafo de 15 linhas.
- O Texto técnico do diagnóstico da Avaliação deve conter apenas o resultado final da avaliação técnica. Não é necessário descrever o passo a passo ou
o raciocínio que foi feito para obter os resultados no texto.
- O Texto técnico do diagnóstico da Avaliação deve informar os instantes iniciais e finais de cada trecho de oscilação que evidencia o agarramento
na válvula de controle.
- O Texto técnico do diagnóstico da Avaliação deve informar qual o formato típico de oscilação que foi identificado e que evidencia a presença de
agarramento/stiction na válvula de controle conforme consta na ORIENTAÇÃO TÉCNICA PARA AVALIAR OSCILAÇÕES PELO EFEITO DE AGARRAMENTO.
- Ao final do Texto técnico do diagnóstico da Avaliação deve estar a conclusão final informando se foi constatado agarramento na válvula de controle ou
se não foi constatado agarramento na válvula de controle.
- O Texto técnico do diagnóstico da Avaliação deve ser gerado sem informar os links dos anexos.


--- Orientações para gerar o PDF no Conde Interpreter ---

Ao gerar o PDF no Code Interpreter:

- Use apenas bibliotecas padrão e não esqueça de import textwrap se for quebrar linhas.
- Garanta que img_path esteja definido e aponte para /mnt/data/grafico_resultado_stiction.png.
- Salve o PDF em /mnt/data/Relatorio_Tecnico_Analise_Malha_<TAG>.pdf.
- Depois de salvar, execute e imprima:
o import os, glob; print('PDFS:', glob.glob('/mnt/data/*.pdf'))
o print('SIZE:', os.path.getsize(pdf_path))
- Se ocorrer exceção, corrija o código e tente novamente até os prints confirmarem.

<==== FIM DA ESPECIFICAÇÃO DAS PREMISSAS PARA ELABORAÇÃO DO RELATÓRIO TÉCNICO DA ANÁLISE


Você irá identificar se há agarramento na malha de controle respectiva a amostra recebida pelo arquivo .csv seguindo o seguinte roteiro:

Para realizar sua tarefa de identificar se há agarramento ou se não há agarramento na malha de controle respectiva a amostra recebida pelo arquivo .csv, execute as
seguintes subtarefas:

1 - Identifique em qual(is) instantes(s) de tempo houveram oscilações significativas
nas formas de onda da PV e na MV. Lembre-se DA ORIENTAÇÃO TÉCNICA PARA DIFERENCIAR
OSCILAÇÕES E TRANSIENTES. 

2 – Avalie se estas oscilações identificadas na PV e na MV obedecem aos padrões típicos
de forma de onda para agarramento, conforme os Critérios obrigatórios para classificar
um trecho como OSCILAÇÃO (limit cycle). Lembre-se DA ORIENTAÇÃO TÉCNICA PARA AVALIAR
OSCILAÇÕES PELO EFEITO DE AGARRAMENTO.

3 – Aplique o Método da Correlação Cruzada para confirmar se estas oscilações
identificadas como padrões típicos de agarramento são realmente causadas por
agarramento na válvula de controle. Lembre-se DA ORIENTAÇÃO TÉCNICA PARA APLICAÇÃO
DO MÉTODO DA CORRELAÇÃO CRUZADA

4 – Gere uma imagem apenas para o gráfico do Resultado da análise da amostra conforme
formatação solicitada. Publicar/anexar a imagem gerada do gráfico como output e na
formatação solicitada. 

5 – Gere o Relatório Técnico da análise conforme modelo solicitado. Publicar/anexar o
arquivo em extensão .pdf com formatação ABNT como output no modelo solicitado.
Responda em texto puro, sem utilizar Markdown. Não inclua linhas com "sandbox:/mnt/
data/..." no texto. Use apenas frases e quebras de linha simples. Lembre-se DA
ESPECIFICAÇÃO DAS PREMISSAS PARA ELABORAÇÃO DO RELATÓRIO TÉCNICO DA ANÁLISE.

6 – Verifique se o Relatório Técnico de análise foi publicado/anexado conforme
solicitado.

"""

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

