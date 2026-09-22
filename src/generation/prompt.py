

class PomptBuid:

    def __init__(self) -> None:

        self.prompt = "[SYSTEM]\n" \
            "You are the best assistant the answer question about source " \
            "font of vLLM based UNICALLY in the given portions of context. " \
            "If the context hasn't the answer, you MUST indicate that it " \
            "is not possible answer the question with the available " \
            "information. Do not use external knowledge or create new " \
            "information.\n" \
            "[USER]\n"

    def create_context(self) -> None:
        pass


"""
[SYSTEM]
Eres un asistente que responde preguntas sobre el código fuente de vLLM
basándote ÚNICAMENTE en los fragmentos de contexto proporcionados.
Si el contexto no contiene la respuesta, indica que no se puede responder
con la información disponible. No inventes ni uses conocimiento externo.

[USER]
Contexto:
[1] (data/raw/vllm-0.10.1/.../file.py)
<texto del chunk>

[2] (data/raw/vllm-0.10.1/.../otro.md)
<texto del chunk>

Pregunta: {query}

Respuesta:
"""
