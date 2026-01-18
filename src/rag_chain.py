from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def get_rag_chain(vector_store, groq_api_key, model_name="llama-3.3-70b-versatile"):
    """
    Creates a RAG chain using LCEL.
    Returns a dict with 'answer' and 'context'.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_store.as_retriever()

    # Create the RAG chain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": itemgetter("input") | retriever, "input": itemgetter("input")}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source
