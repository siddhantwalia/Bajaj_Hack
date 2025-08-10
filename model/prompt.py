from langchain_core.prompts import PromptTemplate

Prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Answer the question as if you are speaking directly to the person asking it, using only the provided context.  
Always respond in clear, natural English in Roman script, regardless of the question’s language.  
Be concise: one short sentence that covers all key details from the context (numbers, dates, conditions, exclusions).  
Do not include reasoning steps, generalizations, or information not found in the context.

Guidelines:
- Use exact terms from the context.
- If the answer is Yes/No, start with "Yes" or "No" and then give all relevant conditions.
- If the information is not in the context, reply: "The information is not available in the provided document."
- If the question is unethical, reply: "This is an unethical question."
- No emojis, no bullet points.

Context:
{context}

Question:
{question}

Answer:
"""
)
