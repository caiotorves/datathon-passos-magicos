# Datathon FIAP — Associação Passos Mágicos

Projeto desenvolvido como parte do Datathon da Pós-Tech FIAP, com foco na análise dos dados PEDE da Associação Passos Mágicos.

## Objetivo

Analisar a evolução dos indicadores educacionais dos alunos entre 2022 e 2024, responder às perguntas de negócio propostas no Datathon e construir um modelo preditivo para identificação de risco de defasagem.

## Sobre os dados

A análise utiliza dados da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE), contemplando indicadores como:

- IAN — Indicador de Adequação de Nível
- IDA — Indicador de Desempenho Acadêmico
- IEG — Indicador de Engajamento
- IAA — Indicador de Autoavaliação
- IPS — Indicador Psicossocial
- IPP — Indicador Psicopedagógico
- IPV — Indicador de Ponto de Virada
- INDE — Índice de Desenvolvimento Educacional

## Etapas do projeto

- Leitura e inspeção inicial das bases
- ETL e padronização longitudinal
- Análise exploratória dos indicadores
- Respostas às perguntas de negócio
- Criação de variáveis para Machine Learning
- Construção e avaliação de modelo preditivo
- Exportação de artefatos para aplicação em Streamlit

## Modelo preditivo

- Algoritmo: Random Forest (200 estimadores, class_weight=balanced)
- Target: aluno em risco (IAN = 2.5 ou IDA < 5.0)
- ROC-AUC: 99.9% | Recall: 96.9% | Precisão: 100.0%

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Conteúdo do repositório

- datathon_passos_magicos.ipynb — Análise completa: ETL, EDA, perguntas de negócio e modelo preditivo
- app.py — Aplicação Streamlit para predição de risco educacional
- modelo.joblib — Modelo Random Forest serializado
- features.joblib — Lista de features do modelo
- requirements.txt — Dependências do projeto
