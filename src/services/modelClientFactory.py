from openai import OpenAI
import os
from groq import Groq

class ModelClientFactory:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
        else:
            self.groq_client = None

        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None

    def get_client(self, provider: str):
        if provider == "groq":
            if not self.groq_client:
                raise Exception("Groq API key no configurada")
            return self.groq_client

        elif provider == "openai":
            if not self.openai_client:
                raise Exception("OpenAI API key no configurada")
            return self.openai_client

        else:
            raise Exception(f"Proveedor no soportado: {provider}")
