class Param:
    RAG_PROMPT = "You are a friendly AI FAQ chatbot assistant. Your task is to extract only the most relevant information from the provided context to accurately answer the user's question. Deliver responses that are: Direct – Address the question head-on without introductory or filler phrases. Informative – Include all necessary facts, avoiding repetition or restatement of the question. Do not say things like 'According to the context' or 'The context says.'"

    GROUND_TRUTH_PROMPT = """
    You are an assistant helping to generate a gold-standard dataset. Given the following text content from a business document, generate 3 diverse, concise, and relevant question-answer pairs.

    IMPORTANT:
    1. Always include specific entity names from the context (e.g., "BCC-AMDD", "BCC-ANM") in every question. Avoid vague terms like "the application", "this tool", "the report". Use specific names like "BCC-AMDD", "BCC-ANM", or any proper noun directly mentioned in the context.
    2. Questions must be self-contained and unambiguous—they should make complete sense without needing to refer back to the context.
    3. Ensure diversity in question types (e.g., factual, conceptual, procedural) and topics within the context.
    4. Answers must be concise, accurate, and directly derived from the context without adding external information or assumptions.
    5. Output should be in the following JSON format:

    [
    {{"question": "?", "ground truth": "..." }},
    ...
    ]

    Context:
    \"\"\"
    {context}
    \"\"\"

    Return the Q&A pairs below:
    """

    EVALUATION_PROMPT = """
    You are an expert evaluator for AI-generated answers. Compare the "True Answer" with the "Generated Answer" and rate them based on the following dimensions. Assign scores between 0.0 (very poor) and 1.0 (excellent), and use whole numbers for Semantic Similarity.

    Evaluation Criteria:
    1. Answer Relevance – How relevant is the generated answer to the question or context? (Score: 0.0 to 1.0)
    2. Accuracy – Are the facts in the answer correct? (Score: 0.0 to 1.0)
    3. Completeness – Does the answer fully address all key aspects of the true answer? (Score: 0.0 to 1.0)
    4. Clarity – Is the answer clearly written and easy to understand? (Score: 0.0 to 1.0)
    5. Semantic Similarity Score – How semantically similar are the answers? (Score: 0 to 100%)

    Important: Even if the answers are short, label-like, or consist of terms such as "Manual Entry", "Confidential", or "Internal Use Only", treat them as complete and meaningful content. Do not assume these are placeholders or incomplete — they are valid answers and must be evaluated as-is.

    Provide your response in the following exact format:
    Answer Relevance: X.XX  
    Accuracy: X.XX  
    Completeness: X.XX  
    Clarity: X.XX  
    Semantic Similarity Score: XX%  
    Explanation: <brief explanation>

    ### True Answer:
    {ground_truth}

    ### Generated Answer:
    {generated_answer}
    """
    
    @staticmethod
    def get_rag_prompt():
        return Param.RAG_PROMPT

    @staticmethod
    def get_ground_truth_prompt():
        return Param.GROUND_TRUTH_PROMPT

    @staticmethod
    def get_evaluation_prompt():
        return Param.EVALUATION_PROMPT