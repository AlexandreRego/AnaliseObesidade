# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Análise Obesidade | Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

PIPELINE_FILENAME = os.path.join("https://github.com/AlexandreRego/AnaliseObesidade/blob/main/src/modelo_rf_pipeline.pkl")
df = pd.read_csv('https://raw.githubusercontent.com/AlexandreRego/AnaliseObesidade/refs/heads/main/data/raw/Obesity.csv')
numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

default_values = {
    "gender": 'Male', "age": 14, "height": 1.40, "weight": 40.0,
    "family_history": 'no', "favc": 'no', "fcvc": 1.0, "ncp": 1.0,
    "caec": 'no', "smoke": 'no', "ch2o": 1.0, "scc": 'no',
    "faf": 0.0, "tue": 0.0, "calc": 'no', "mtrans": 'Automobile'
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

def limpar_filtros():
    for key, value in default_values.items():
        st.session_state[key] = value

def train_and_save_artifacts():
    with st.spinner('Treinando o modelo... Isso pode levar um momento.'):
        try:
            dados = df.copy()
        except Exception as e: 
            st.error(f"Erro ao carregar o arquivo de dados do GitHub: {e}")
            return

        X = dados.drop('Obesity', axis=1)
        y = dados['Obesity']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4532678, stratify=y)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        modelo_rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('classifier', RandomForestClassifier(random_state=4532678))])
        
        modelo_rf_pipeline.fit(X_train, y_train)

        y_pred = modelo_rf_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f'Modelo treinado! Precisão no conjunto de teste: {accuracy * 100:.2f}%')

        joblib.dump(modelo_rf_pipeline, PIPELINE_FILENAME)
        st.success(f"Pipeline do modelo salvo com sucesso como '{PIPELINE_FILENAME}'!")

@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load(PIPELINE_FILENAME)
        return pipeline 
    except FileNotFoundError:
        return None

@st.cache_data
def load_raw_data():
    try:
        return df.copy()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_resource
def gerar_pairplot(df):
    return sns.pairplot(
        df,
        hue="Obesity",
        palette="tab10",
        corner=True,
        plot_kws={'s': 10}  
    )
    
st.title("Dashboard de Análise de Risco de Obesidade")

_, col_img, _ = st.columns([1, 4, 1])
caminho_imagem = "https://raw.githubusercontent.com/AlexandreRego/AnaliseObesidade/refs/heads/main/data/raw/tabela-imc_202108231741.png"

with col_img:
    st.image(
        caminho_imagem,
        caption="Tabela de Classificação do Índice de Massa Corporal (IMC)",
        use_container_width=True
    )

st.markdown("---")

    
opcoes_genero = {"Male": "Masculino", "Female": "Feminino"}
opcoes_sim_nao = {'yes': "Sim", 'no': "Não"}
opcoes_caec_calc = {'no': "Não", 'Sometimes': "De vez em quando", 'Frequently': "Com frequência", 'Always': "Diariamente"}
opcoes_mtrans = {'Automobile': "Automóvel", 'Motorbike': "Moto", 'Bike': "Bicicleta", 'Public_Transportation': "Transporte Público", 'Walking': "A pé"}

st.sidebar.header("Controles do Formulário")
st.sidebar.button("🧹 Limpar Filtros", on_click=limpar_filtros, use_container_width=True)

st.sidebar.header("📊 Insira seus Dados para Análise")

with st.sidebar.form("formulario_previsao"):
    st.subheader("👤 Informações Pessoais")
    
    col_genero, col_idade = st.columns(2)
    with col_genero:
        st.selectbox("Gênero", options=opcoes_genero.keys(), format_func=lambda x: opcoes_genero[x], key="gender")
    with col_idade:
        st.slider("Idade", 14, 70, key="age")

    col_altura, col_peso = st.columns(2)
    with col_altura:
        st.slider("Altura (m)", 1.40, 2.10, format="%.2f", key="height")
    with col_peso:
        st.slider("Peso (kg)", 40.0, 180.0, format="%.1f", key="weight")

    st.subheader("🍏 Hábitos Alimentares e Estilo de Vida")
    st.selectbox("Histórico familiar de sobrepeso?", options=list(opcoes_sim_nao.keys()), format_func=lambda x: opcoes_sim_nao[x], key="family_history")
    st.selectbox("Consumo de alimentos calóricos (FAVC)?", options=list(opcoes_sim_nao.keys()), format_func=lambda x: opcoes_sim_nao[x], key="favc")
    st.slider("Consumo de vegetais (FCVC)", 1.0, 3.0, step=0.5, help="1: Nunca, 2: Às vezes, 3: Sempre", key="fcvc") 
    st.slider("Refeições principais por dia", 1.0, 4.0, step=0.5, key="ncp") 
    st.selectbox("Consome lanches entre as refeições?", options=list(opcoes_caec_calc.keys()), format_func=lambda x: opcoes_caec_calc[x], key="caec")
    st.selectbox("É fumante?", options=list(opcoes_sim_nao.keys()), format_func=lambda x: opcoes_sim_nao[x], key="smoke")
    st.slider("Consumo diário de água (Litros)", 1.0, 4.0, step=0.5, key="ch2o")
    st.selectbox("Monitora o consumo de calorias?", options=list(opcoes_sim_nao.keys()), format_func=lambda x: opcoes_sim_nao[x], key="scc")
    st.slider("Frequência de atividade física (dias/semana)", 0.0, 7.0, step=0.5, key="faf") 
    st.slider("Tempo de uso de telas (horas/dia)", 0.0, 5.0, step=0.1, key="tue") 
    st.selectbox("Frequência no consumo de álcool (CALC)", options=list(opcoes_caec_calc.keys()), format_func=lambda x: opcoes_caec_calc[x], key="calc")
    st.selectbox("Meio de transporte", options=list(opcoes_mtrans.keys()), format_func=lambda x: opcoes_mtrans[x], key="mtrans")

    st.markdown("---")
    botao_submeter = st.form_submit_button("✨ Gerar Previsão")

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Treinar Modelo com CSV do GitHub"):
    train_and_save_artifacts()

modelo = load_artifacts() 
dados_brutos = load_raw_data()

if not botao_submeter:
    st.info("Utilize a barra lateral para inserir seus dados e clique em 'Gerar Previsão' para obter uma análise completa.")
else:

    if modelo is not None: 
        input_dict = {
            "Gender": st.session_state.gender, "Age": st.session_state.age, "Height": st.session_state.height, "Weight": st.session_state.weight,
            "family_history": st.session_state.family_history, "FAVC": st.session_state.favc, "FCVC": st.session_state.fcvc, "NCP": st.session_state.ncp,
            "CAEC": st.session_state.caec, "SMOKE": st.session_state.smoke, "CH2O": st.session_state.ch2o, "SCC": st.session_state.scc, "FAF": st.session_state.faf,
            "TUE": st.session_state.tue, "CALC": st.session_state.calc, "MTRANS": st.session_state.mtrans
        }
        input_df = pd.DataFrame([input_dict])
        predicao = modelo.predict(input_df)[0] 
        probabilidade_predicao = modelo.predict_proba(input_df)
        imc = st.session_state.weight / (st.session_state.height ** 2)

        st.header("🎯 Resultado da Análise Preditiva")
        col_metrica1, col_metrica2 = st.columns(2)
        with col_metrica1:
            st.metric(label="Nível de Obesidade Previsto", value=str(predicao).replace("_", " "))
        with col_metrica2:
            st.metric(label="Seu IMC (Índice de Massa Corporal)", value=f"{imc:.2f}")

        if "Normal_Weight" in predicao:
            st.balloons()
            st.success(f"🎉 Ótima notícia! O modelo indica uma forte tendência para **Peso Normal**. Continue com seus hábitos saudáveis!")
        elif "Insufficient_Weight" in predicao:
            st.warning(f"⚠️ Atenção: O modelo indica uma tendência para **Peso Insuficiente**. É importante buscar orientação profissional.")
        elif "Overweight" in predicao:
            st.warning(f"⚠️ Atenção: O modelo indica uma tendência para **Sobrepeso**. Considerar ajustes nos hábitos pode ser benéfico.")
        elif "Obesity" in predicao:
            st.error(f"🚨 Alerta: O modelo indica uma forte tendência para **Obesidade**. É altamente recomendado procurar acompanhamento médico.")

        st.subheader("Probabilidade por Categoria")
        df_prob = pd.DataFrame(probabilidade_predicao, columns=modelo.classes_, index=["Probabilidade"]).T
        df_prob.index = df_prob.index.str.replace('_', ' ')
        df_prob = df_prob.sort_values(by="Probabilidade", ascending=False)
        fig_prob = px.bar(
            df_prob, x=df_prob.index, y='Probabilidade',
            labels={'Probabilidade': 'Probabilidade', 'index': 'Nível de Obesidade'},
            text_auto='.2%', title="Probabilidade para Cada Nível de Obesidade"
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    else:
        st.error("O pipeline do modelo não foi carregado corretamente. Por favor, treine o modelo usando o botão na barra lateral.")

if dados_brutos is not None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("📊 Painel Analítico Interativo")

    with st.expander("Distribuição Geral e Fatores Relevantes", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribuição Geral da Obesidade")
            contagem_obesidade = dados_brutos['Obesity'].value_counts()
            fig_pizza = px.pie(names=contagem_obesidade.index, values=contagem_obesidade.values,
                                title="Proporção de cada Nível de Obesidade", hole=0.3)
            st.plotly_chart(fig_pizza, use_container_width=True)
        with col2:
            st.subheader("Fatores Mais Relevantes na Previsão")
            
            if modelo is not None and hasattr(modelo.named_steps['classifier'], 'feature_importances_'):
                try:
                    importances = modelo.named_steps['classifier'].feature_importances_
                    feature_names_out = modelo.named_steps['preprocessor'].get_feature_names_out()
                    df_importancia = pd.DataFrame({'Fator': feature_names_out, 'Importância': importances})
                    df_importancia['Fator'] = df_importancia['Fator'].str.replace('num__', '').str.replace('cat__', '').str.replace('remainder__', '').str.replace('_', ' ')
                    df_importancia = df_importancia.sort_values(by='Importância', ascending=False).head(15)
                    fig_importancia = px.bar(
                        df_importancia, x='Importância', y='Fator', orientation='h',
                        title='Importância de Cada Fator para o Modelo',
                        labels={'Importância': 'Importância Relativa', 'Fator': 'Fator'}
                    )
                    fig_importancia.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_importancia, use_container_width=True)
                except Exception as e:
                    st.warning(f"Não foi possível exibir a importância dos fatores. Erro: {e}")
            else:
                st.warning("Não foi possível exibir a importância dos fatores. Verifique se o modelo pipeline foi carregado e se o classificador possui o atributo 'feature_importances_'.")
                
                
    with st.expander("📌 Relação Cruzada entre Variáveis (Pairplot)", expanded=False):
        st.markdown("Selecione até 4 variáveis numéricas para explorar a relação entre elas:")

        selected_vars = st.multiselect(
            "Variáveis para análise",
            numeric_features,
            default=numeric_features[:3],  
            max_selections=4
        )

        if len(selected_vars) >= 2:
            df_pairplot = dados_brutos[selected_vars + ['Obesity']].copy()
            df_sample = df_pairplot.sample(n=min(300, len(df_pairplot)), random_state=42)
            fig_pairplot = sns.pairplot(df_sample, hue="Obesity", palette="tab10", corner=True, plot_kws={'s':10})
            st.pyplot(fig_pairplot)
        else:
            st.info("Selecione pelo menos 2 variáveis para gerar o pairplot.")


    with st.expander("Análise Demográfica e de Hábitos"):
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Distribuição de Idade por Nível de Obesidade")
            fig_boxplot_idade = px.box(dados_brutos, x="Obesity", y="Age", color="Obesity",
                                         title="Faixa Etária por Categoria de Obesidade",
                                         labels={"Obesity": "Nível de Obesidade", "Age": "Idade"})
            st.plotly_chart(fig_boxplot_idade, use_container_width=True)
        with col4:
            st.subheader("Distribuição de Obesidade por Gênero")
            obesidade_genero = dados_brutos.groupby(['Gender', 'Obesity']).size().reset_index(name='Contagem')
            fig_barras_genero = px.bar(
                obesidade_genero, x="Obesity", y="Contagem", color="Gender", barmode="group",
                title="Contagem de Níveis de Obesidade por Gênero",
                labels={"Obesity": "Nível de Obesidade", "Contagem": "Número de Pessoas", "Gender": "Gênero"}
            )
            st.plotly_chart(fig_barras_genero, use_container_width=True)

    with st.expander("Relações entre Fatores Físicos e Obesidade"):
        st.subheader("Relação entre Peso, Altura e Nível de Obesidade")
        fig_dispersao = px.scatter(
            dados_brutos, x="Height", y="Weight", color="Obesity",
            hover_data=['Age', 'Gender'], title="Dispersão de Peso vs. Altura",
            labels={"Height": "Altura (m)", "Weight": "Peso (kg)", "Obesity": "Nível de Obesidade"}
        )
        st.plotly_chart(fig_dispersao, use_container_width=True)

        st.subheader("Matriz de Correlação entre Variáveis Numéricas")
        st.markdown("Esta matriz mostra como as variáveis numéricas se relacionam. Valores próximos de 1 (vermelho) ou -1 (azul) indicam forte correlação.")
        df_numerico = dados_brutos.select_dtypes(include=[np.number])
        correlacao = df_numerico.corr()

        fig_heatmap = px.imshow(
            correlacao, text_auto=True, aspect="auto",
            color_continuous_scale='RdBu_r', title="Mapa de Calor das Correlações"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    st.error("Os dados brutos não foram carregados corretamente para o painel analítico.")
