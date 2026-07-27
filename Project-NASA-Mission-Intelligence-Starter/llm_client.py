from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    system_prompt = """You are an expert NASA mission specialist with deep knowledge of space exploration history. \
You have access to official NASA mission documents, transcripts, and technical reports from Apollo 11, Apollo 13, and the Challenger missions.

Your role is to:
- Answer questions about NASA space missions accurately and in detail
- Cite specific sources from the provided context when answering
- Acknowledge clearly when the provided context does not contain enough information to answer confidently
- Provide historical context and technical details when relevant

When answering:
- Always reference the source documents you are drawing from
- Be precise with dates, crew names, and technical details
- If the context is insufficient, say so explicitly rather than speculating
- Maintain a professional, informative tone"""

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Use the following NASA mission documents as context for your answer:\n\n{context}"
        })

    for turn in conversation_history:
        messages.append(turn)

    messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content