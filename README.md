# Tech Challenge – Fase 04: Previsão de Obesidade

## Descrição do Projeto  
Este repositório contém a solução desenvolvida para o Tech Challenge Fase 04 – Data Analytics, cujo objetivo foi construir um modelo de Machine Learning capaz de prever o nível de obesidade de uma pessoa a partir de um conjunto de variáveis sócio-demográficas e de hábitos de vida. Além da pipeline de modelagem, inclui-se uma aplicação preditiva em Streamlit e um painel analítico com insights sobre obesidade.

## Funcionalidades  
- **Pipeline de Machine Learning**  
  - Carregamento e limpeza do conjunto de dados (`obesity.csv`)  
  - Feature engineering (tratamento de variáveis categóricas, normalização, geração de novas features)  
  - Treinamento e validação de modelos  
  - Seleção de modelo com assertividade ≥ 75%  

- **Aplicação preditiva (Streamlit)**  
  - Interface web para inserção de dados do paciente  
  - Retorno do nível de obesidade previsto pelo modelo  

- **Dashboard analítico**  
  - Gráficos e tabelas com principais insights (distribuição de IMC, correlações, perfil de risco)  
  - Indicadores estatísticos para auxiliar a equipe médica  

- **Containerização (Docker)**  
  - Imagem Docker com todas as dependências instaladas  
  - Fácil deployment em qualquer ambiente  

## Tecnologias e Bibliotecas  
- **Linguagem:** Python 3.8+  
- **Manipulação de dados:** pandas, numpy  
- **Machine Learning:** scikit-learn  
- **Visualização:** matplotlib, seaborn, plotly  
- **Web App:** Streamlit  
- **Containerização:** Docker, Docker Compose

## Plataforma da aplicação em Streamlit para consulta e simulações

- Streamlit: https://alexandrerego-analiseobesidade-srcmodelo-obs-w115vv.streamlit.app/
