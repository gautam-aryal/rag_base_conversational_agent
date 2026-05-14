from langchain.prompts import PromptTemplate


class Augmentation:

  def __init__(self,):
    pass

  def prompt_for_llama3(self, query, contexts):
    # LLaMA 3 chat-style prompt template
    base_system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant in a retrieval-augmented generation (RAG) system.
    Use the following context to answer the user's question as accurately and informatively as possible.
    Do not write any pre-fix befor starting the answer such as 'According to the provided context'; directly start answer after Answer: 

    Context:
    {context}

    <|start_header_id|>user<|end_header_id|>
    {query}

    <|start_header_id|>assistant<|end_header_id|>"""

    # Build LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=base_system_prompt
    )

    # Format prompt with retrieved context
    prompt = prompt_template.format(
        query=query,
        context=Augmentation.doc_to_text(contexts)
    )

    return prompt

  def prompt_for_mistral(self):
    pass

  def prompt_for_gimma3_12b(self, query, contexts):
    # RAG-style system prompt template
    base_system_prompt = """<|start|>system<|message|>
    You are a helpful assistant in a retrieval-augmented generation (RAG) system.
    Do not write any pre-fix befor starting the answer such as 'According to the provided context'; directly start answer after Answer: 
    Context:
    {context}
    <|end|>

    <|start|>user<|message|>
    Question: {query}
    <|end|>

    <|start|>assistant<|message|>"""

    # Build LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=base_system_prompt
    )

    # Example input (normally you'd inject retrieved docs here)
    prompt = prompt_template.format(
        query = query,
        context = Augmentation.doc_to_text(contexts)
    )

    return prompt

  def prompt_for_gpt_oss_20b(self, query, contexts):
    # RAG-style system prompt template
    base_system_prompt = """<|start|>system<|message|>
    You are a helpful assistant in a retrieval-augmented generation (RAG) system.
    Do not write any pre-fix befor starting the answer such as 'According to the provided context'; directly start answer after Answer: 
    Knowledge cutoff: 2024-06
    Current date: {{ currentDate }}

    Reasoning: none
    Web access: disabled
    Tools: disabled

    Instruction:
    - Use ONLY the provided context to answer the user’s question.
    - If the context is irrelevant, incomplete, or does not contain the answer, reply with:
      "I don’t know based on the given context."
    - Do NOT make up facts or provide external knowledge.
    <|end|>

    <|start|>user<|message|>
    Question: {query}

    Context:
    {context}
    <|end|>

    <|start|>assistant<|message|>"""

    # Build LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=base_system_prompt
    )

    # Example input (normally you'd inject retrieved docs here)
    prompt = prompt_template.format(
        query = query,
        context = Augmentation.doc_to_text(contexts)
    )

    return prompt

  def doc_to_text(contexts):
    return contexts
    # if contexts is None:
    #   return "None"

    # return ", ".join([ str(i + 1) + "." + context.page_content for i, context in enumerate(contexts)])


class InContextAugmentation:
  def __init__(self,):
    pass

  def prompt_for_llama3(self, query, contexts):
    # LLaMA 3 chat-style prompt template
    base_system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant in a retrieval-augmented generation (RAG) system.
    Use the following context to answer the user's question as accurately and informatively as possible.
    Do not write any pre-fix befor starting the answer such as 'According to the provided context'; directly start answer after Answer: 
    
    Few-shot-example:  
    Q1: "What are the main functions of Nepal Rastra Bank?"
    A1: "The main functions of Nepal Rastra Bank are issuing Nepalese currency and coins, licensing and regulating banks and financial institutions, formulating monetary and foreign exchange policies, managing foreign exchange reserves, acting as banker and financial advisor to the Government of Nepal, promoting and regulating payment and settlement systems, and managing liquidity." 
    Q2: "How many types of payment cards are prevalent in Nepal?"
    A2: "There are currently three types of payment cards prevalent in Nepal: credit cards, debit cards, and prepaid cards."
    Q3: "Is accounting for investments and loans mandatory? 
    A3: "Yes, accounting for foreign investment and foreign loans is mandatory in Nepal to repatriate unaccounted investment and loan proceeds and to buy or sell shares held in the name of foreign investors.
    Q4: "What was the y-o-y (y-o-y) expansion of M2 in mid-September 2024?"
    A4: "M2 expanded 13.9 percent y-o-y."

    Context:
    {context}

    <|start_header_id|>user<|end_header_id|>
    {query}

    <|start_header_id|>assistant<|end_header_id|>"""

    # Build LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=base_system_prompt
    )

    # Format prompt with retrieved context
    prompt = prompt_template.format(
        query=query,
        context=Augmentation.doc_to_text(contexts)
    )

    return prompt

  def prompt_for_mistral(self):
    pass

  def prompt_for_gimma3_12b(self, query, contexts):
    # RAG-style system prompt template
    base_system_prompt = """<|start|>system<|message|>
    You are a helpful assistant in a retrieval-augmented generation (RAG) system.
    Do not write any pre-fix befor starting the answer such as 'According to the provided context'; directly start answer after Answer: 
    Few-shot-example:  
    Q1: "What are the main functions of Nepal Rastra Bank?"
    A1: "The main functions of Nepal Rastra Bank are issuing Nepalese currency and coins, licensing and regulating banks and financial institutions, formulating monetary and foreign exchange policies, managing foreign exchange reserves, acting as banker and financial advisor to the Government of Nepal, promoting and regulating payment and settlement systems, and managing liquidity." 
    Q2: "How many types of payment cards are prevalent in Nepal?"
    A2: "There are currently three types of payment cards prevalent in Nepal: credit cards, debit cards, and prepaid cards."
    Q3: "Is accounting for investments and loans mandatory? 
    A3: "Yes, accounting for foreign investment and foreign loans is mandatory in Nepal to repatriate unaccounted investment and loan proceeds and to buy or sell shares held in the name of foreign investors.
    Q4: "What was the y-o-y (y-o-y) expansion of M2 in mid-September 2024?"
    A4: "M2 expanded 13.9 percent y-o-y."

    Context:
    {context}
    <|end|>
    
    <|start|>user<|message|>
    Question: {query}
    <|end|>

    <|start|>assistant<|message|>"""

    # Build LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=base_system_prompt
    )

    # Example input (normally you'd inject retrieved docs here)
    prompt = prompt_template.format(
        query = query,
        context = Augmentation.doc_to_text(contexts)
    )

    return prompt

  def prompt_for_gpt_oss_20b(self, query, contexts):
    base_system_prompt = """<|start|>system<|message|>
    You are a helpful assistant in a retrieval-augmented generation (RAG) system.

    Guidelines:
    - Use ONLY the provided context to answer the user’s question.
    - Do NOT add prefixes like "According to the context" — start directly with the answer.
    - If the context does not contain the answer, reply: "I don’t know based on the given context."
    - Do NOT invent facts or use external knowledge.
    - Keep answers concise, factual, and well-structured.

    Knowledge cutoff: 2024-06
    Current date: {{ currentDate }}
    Reasoning: none
    Web access: disabled
    Tools: disabled
    <|end|>

    <|start|>user<|message|>
    Question: {query}

    Examples:
    Q1: "What are the main functions of Nepal Rastra Bank?"
    A1: "Issuing currency, regulating banks, formulating monetary policy, managing reserves, advising government, regulating payments, managing liquidity."

    Q2: "How many types of payment cards are prevalent in Nepal?"
    A2: "Three types: credit cards, debit cards, and prepaid cards."

    Q3: "Is accounting for investments and loans mandatory?"
    A3: "Yes, accounting for foreign investment and loans is mandatory in Nepal to repatriate unaccounted proceeds and manage share transactions."

    Q4: "What was the y-o-y expansion of M2 in mid-September 2024?"
    A4: "M2 expanded 13.9 percent y-o-y."

    Context:
    {context}
    <|end|>

    <|start|>assistant<|message|>"""

    # Build LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=base_system_prompt
    )

    # Example input (normally you'd inject retrieved docs here)
    prompt = prompt_template.format(
        query = query,
        context = Augmentation.doc_to_text(contexts)
    )

    return prompt

  def doc_to_text(contexts):
    return contexts