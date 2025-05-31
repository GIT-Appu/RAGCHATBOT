import streamlit as st

try:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    import os

    from dotenv import load_dotenv
    load_dotenv()

    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
    db_name = r"health_supplements"
    vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOllama(model='gemma3:1b', base_url='http://localhost:11434')

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    error = None

except Exception as e:
    error = str(e)

# Streamlit UI
st.title("RAG Chatbot with Ollama & LangChain")
st.write("Ask a question about health supplements:")

if error:
    st.error(f"Initialization error: {error}")
else:
    user_input = st.text_input("Your question", "")

    if st.button("Ask") and user_input.strip():
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)
            st.markdown("**Answer:**")
            st.write(response)