import os
import openai
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

# Configuração da chave da OpenAI
openai.api_key = "codigo-api"

# Carregar PDF e extrair texto
def carregar_documentos(caminho_pdf):
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    
    # Dividir o texto em segmentos menores
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    textos_segmentados = splitter.split_documents(documentos)
    
    return textos_segmentados

# Criar e armazenar embeddings no banco vetorial
def criar_base_vetorial(textos_segmentados):
    persist_directory = "./db"
    
    # Configurar embeddings da OpenAI
    embeddings = OpenAIEmbeddings()
    
    # Criar ou carregar banco vetorial
    vectorstore = Chroma.from_documents(
        documentos=textos_segmentados, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    return vectorstore

# Criar chatbot para responder perguntas
def responder_pergunta(pergunta, vectorstore):
    # Buscar os trechos mais relevantes
    documentos_relevantes = vectorstore.similarity_search(pergunta, k=3)
    
    # Concatenar os trechos encontrados
    contexto = "\n\n".join([doc.page_content for doc in documentos_relevantes])
    
    # Gerar resposta usando OpenAI GPT
    resposta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um assistente especializado em responder perguntas com base em documentos PDF fornecidos."},
            {"role": "user", "content": f"Aqui está um contexto do PDF:\n\n{contexto}\n\nAgora, responda à seguinte pergunta: {pergunta}"}
        ]
    )
    
    return resposta["choices"][0]["message"]["content"]

# Testando o chatbot
if __name__ == "__main__":
    caminho_pdf = "exemplo.pdf"  # Nome do arquivo PDF de exemplo

    print("Carregando documentos...")
    documentos = carregar_documentos(caminho_pdf)
    
    print("Criando banco vetorial...")
    vectorstore = criar_base_vetorial(documentos)
    
    while True:
        pergunta = input("\nFaça uma pergunta (ou digite 'sair' para encerrar): ")
        if pergunta.lower() == "sair":
            break
        
        resposta = responder_pergunta(pergunta, vectorstore)
        print("\nResposta:", resposta)
