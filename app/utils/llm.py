from langchain.chat_models import init_chat_model   # type: ignore
from langchain.chains import ConversationChain  # type: ignore
from langchain.memory import ConversationBufferMemory   # type: ignore
from langchain_core.messages import SystemMessage, trim_messages, AIMessage, HumanMessage  # type: ignore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder   # type: ignore
import base64
import streamlit as st
import logging


class OCR_LLM:
    # google_llm_model = 'gemini-1.5-flash'
    google_llm_model = 'gemini-2.0-flash'
    openai_llm_model = 'gpt-4.1-nano'

    def __init__(self, openai_api_key=None, google_api_key=None, context: str = ''):
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.context = context if context else "converting written text to digital format"
        self.history_messages = [SystemMessage(
            content=f"You are an OCR assistant expert in {context}. Your task is to analyze images and extract text from them in markdown format. Do not provide any additional information or context. Just focus on the text extraction. If you already have extracted that text, do not repeat it. If you cannot extract any text, output [IMAGE HERE] and do not provide any text.",)]
        self.chosen_model = None
        self.llm = self._initialize_model()
        self.trimmer = trim_messages(
            max_tokens=1000,
            strategy="last",
            token_counter=self.chosen_model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

    def _initialize_model(self):
        if self.google_api_key:
            self.chosen_model = self.google_llm_model
            return init_chat_model(self.google_llm_model, model_provider="google_genai", api_key=self.google_api_key)
        elif self.openai_api_key:
            self.chosen_model = self.openai_llm_model
            return init_chat_model(self.openai_llm_model, model_provider="openai", api_key=self.openai_api_key)
        else:
            raise ValueError("No API key provided. Please set either OpenAI or Google API key.")

    def __call__(self, image_path):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
            new_message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                # {
                #     "type": "text",
                #     "text": "Convert this picture to text in markdown format."
                # }
            ])
            self.history_messages.append(new_message)
        response = self.llm.invoke(self.history_messages)
        if isinstance(response, AIMessage):
            self.history_messages.append(response)
            logging.debug(f"Tokens used: {response.usage_metadata['total_tokens'] if hasattr(response, 'usage_metadata') and response.usage_metadata else 'N/A'}")
            return response
        else:
            logging.error(
                "Error in OCR_LLM call: response is not an AIMessage.")
            raise ValueError(
                "Error in OCR_LLM call: response is not an AIMessage. Please check the API keys and the model configuration."
            )

    def refine(self, ocr_list):
        prompt: str = 'An OCR assistant has extracted the following text from images. Knowing the context of the images, please refine the text to make it more readable and structured. The context is: ' + self.context + '\n\n' # TODO: Improve this prompt
        for i, ocr in enumerate(ocr_list):
            prompt += f"Image {i+1}:\n{ocr}\n\n"
        prompt += "Please refine the text and return it in markdown format."
        self.history_messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(self.history_messages)
        if isinstance(response, AIMessage):
            self.history_messages.append(response)
            return response.content
        else:
            logging.error(
                "Error in OCR_LLM refine: response is not an AIMessage.")
            raise ValueError(
                "Error in OCR_LLM refine: response is not an AIMessage. Please check the API keys and the model configuration."
            )
