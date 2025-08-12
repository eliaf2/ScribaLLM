from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import base64
from PIL import Image
from io import BytesIO
import os
import streamlit as st
import logging


class ImageClassificationStructure(BaseModel):
    classification: Literal["text", "picture"] = Field(
        description="Classification of the image content"
    )
    description: str = Field(
        description="Brief description of the image content in maximum 20 words"
    )


class ImageOut(BaseModel):
    path: str = Field(
        description="Path to the image."
    )
    description: str = Field(
        description="Description of the image."
    )


class AgentState(TypedDict):
    page_b64: str  # Base64 representation of the input page image
    pictures_folder: str
    context: str
    messages: list[HumanMessage | AIMessage | ToolMessage | SystemMessage]
    text: str
    list_pictures: list[ImageOut]
# =======================================


class OCR_LLM:
    # gemini_llm_model = 'gemini-1.5-flash'
    gemini_llm_model = 'gemini-2.0-flash'
    openai_llm_model = 'gpt-4.1-nano'
    temperature = 0.2

    def __init__(self, openai_api_key=None, gemini_api_key=None, context: str = ''):
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.context = context if context else "converting written text to digital format"

        if not self.openai_api_key and not self.gemini_api_key:
            raise ValueError(
                "No API key provided. Please set either OpenAI or Gemini API key.")
        elif self.gemini_api_key:
            self.llm = init_chat_model(model=self.gemini_llm_model, model_provider="google_genai",
                                       api_key=self.gemini_api_key, temperature=self.temperature)
        elif self.openai_api_key:
            self.llm = init_chat_model(model=self.openai_llm_model, model_provider="openai",
                                       api_key=self.openai_api_key, temperature=self.temperature)

        self.classifier_llm = self.llm.with_structured_output(
            ImageClassificationStructure)

        self.graph = StateGraph(AgentState)

        self.graph.add_node("classification", self.llm_classifier)
        self.graph.set_entry_point("classification")
        self.graph.add_node("ocr", self.ocr_function)

        self.graph.add_edge("classification", "ocr")
        self.graph.add_edge("ocr", END)

        self.app = self.graph.compile()

    # ======== Methods for the graph ========
    @staticmethod
    def compress_image(
        input_path: str,
        size_scale: float = 0.3,
        quality: int = 10,
        dpi: int = 10,
        prefix: str = "compressed_",
        save_to_disk: bool = False
    ) -> Image.Image:
        '''Compress the input image

        Parameters
        ----------
        input_path : str
            Path to the input image file.
        size_scale : float, optional
            Scaling factor for resizing the image, by default 0.3
        quality : int, optional
            JPEG quality setting, by default 10
        dpi : int, optional
            DPI setting for the output image, by default 10
        prefix : str, optional
            Prefix for the output file name, by default "compressed_"
        save_to_disk : bool, optional
            Whether to save the compressed image to disk, by default False

        Returns
        -------
        Image.Image
            The compressed image as a PIL Image object.
        '''
        from PIL import Image
        import os
        from io import BytesIO

        img = Image.open(input_path)
        img = img.convert('RGB')
        img = img.resize(
            (int(img.size[0] * size_scale), int(img.size[1] * size_scale)))
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality, dpi=(dpi, dpi))
        buffer.seek(0)
        if save_to_disk:
            dir_name = os.path.dirname(input_path)
            base_name = os.path.basename(input_path)
            new_name = os.path.join(dir_name, f"{prefix}{base_name}")
            with open(new_name, 'wb') as f:
                f.write(buffer.getvalue())
            logging.debug(f"Compressed image saved as: {new_name}")
        new_img = Image.open(buffer)
        return new_img

    def llm_classifier(self, state: AgentState) -> AgentState:
        '''Classify the pictures in the ```pictures_folder``` either as text or picture.

        Parameters
        ----------
        state : AgentState
            The state containing context and images information.

        Returns
        -------
        AgentState
            The updated state after classification.
        '''

        system_prompt = f"""Analyze the provided image and classify it based on its content.

        Classification Rules:
        - Return 'text' if the image contains:
        * Any readable text (handwritten or printed)
        * Mathematical formulas or equations  
        * Code snippets
        * Any form of written content or symbols
        
        - Return 'picture' if the image contains:
        * Charts/graphs with text labels
        * Artwork, drawings, or illustrations without text
        * Screenshots of documents

        Focus on whether there is meaningful text or textual information present in the image.

        Additionally, provide a brief description of what you see in the image using maximum 10 words considering the following context: {state['context']}"""

        for image_path in os.listdir(state['pictures_folder']):
            image_full_path = os.path.join(
                state['pictures_folder'], image_path)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                with BytesIO() as buffer:
                    self.compress_image(image_full_path).save(
                        buffer, format="JPEG")
                    image_b64_compressed = base64.b64encode(
                        buffer.getvalue()).decode("utf-8")

                new_message = HumanMessage(content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64_compressed}"
                        }
                    }
                ])

                response = self.classifier_llm.invoke(
                    [SystemMessage(content=system_prompt)] + [new_message]
                )

                if response and response.classification:    # type: ignore
                    st.write(
                        f"Image: {image_path} classified as {response.classification}. It talks about {response.description}") # type: ignore

                    if response.classification == "picture":    # type: ignore
                        state['list_pictures'].append(ImageOut(
                            path=image_full_path, description=response.description))    # type: ignore

        return state

    @staticmethod
    def create_prompt_with_images(state: AgentState) -> str:
        '''Creates a prompt string that includes the base prompt and image insertion instructions.

        Parameters
        ----------
        state : AgentState
            The state containing context and images information.

        Returns
        -------
        str
            The complete prompt string for the LLM.
        '''

        base_prompt = f"""You are a document-to-markdown converter.  
            Your task is to read the text content of a given PDF page and produce clean, well-structured Markdown output. The PDF talks about {state['context']}.

            Rules:
            1. Ignore all images, plots, or figures — do not include them in the output.
            2. Preserve headings, bullet points, numbered lists, tables, and code blocks in proper Markdown syntax.
            3. Convert any mathematical formulas to LaTeX math notation:
            - Inline math: wrap formulas in single dollar signs `$ ... $`.
            - Display math: wrap block formulas in double dollar signs `$$ ... $$` on separate lines.
            4. Preserve paragraph breaks and indentation appropriately so that the output can be pasted directly into a `.md` file and render correctly in Markdown viewers.
            5. Remove any artifacts from OCR or PDF conversion (e.g., hyphenated line breaks, random page numbers, headers/footers not part of the main text).
            6. Output only the Markdown text — no explanations, comments, or metadata.
            """

        if state['list_pictures']:
            image_instruction = f"""
            7. Insert the following images at appropriate locations in the document using standard Markdown image syntax:
            Available images:"""

            for image in state['list_pictures']:
                image_instruction += f"\n        * {image.path}: {image.description}"

            image_instruction += """\n        Use this format: ![alt text](image_filename)
            - Choose descriptive alt text based on the surrounding content
            - Place images where they would logically fit in the document flow
            - Only use images from the provided list"""

            base_prompt += image_instruction

        example = """
            Example:
            PDF text:
            The equation of motion is x(t) = x_0 + v_0 t + (1/2) a t^2.

            Markdown output:
            The equation of motion is $x(t) = x_0 + v_0 t + \\frac{{1}}{{2}} a t^2$."""

        return base_prompt + example

    def ocr_function(self, state: AgentState) -> dict:
        '''Perform OCR on the images in the provided state.

        Parameters
        ----------
        state : AgentState
            The state containing context and images information.

        Returns
        -------
        dict
            The OCR results for the images.
        '''
        system_prompt = self.create_prompt_with_images(state)
        response = self.llm.invoke(
            [SystemMessage(content=system_prompt)] + list(state['messages'])[-1:])
        return {
            'text': response.content
        }

    # =======================================

    def __call__(self, page_image_path: str, pictures_folder: str) -> tuple[list, list[ImageOut]]:
        '''Perform OCR on a single page image and its associated pictures.

        Parameters
        ----------
        page_image_path : str
            Path to the page image file.
        pictures_folder : str
            Path to the folder containing pictures related to the page.

        Returns
        -------
        tuple[list, list[ImageOut]]
            A tuple containing the OCR results and the list of pictures as ```ImageOut``` objects.
        '''
        with open(page_image_path, "rb") as f:
            page_b64 = base64.b64encode(f.read()).decode("utf-8")
            new_message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{page_b64}"
                    }
                }
            ])
        call = self.app.invoke({'page_b64': page_b64,
                                'pictures_folder': pictures_folder,
                                'context': self.context,
                                'list_pictures': [],
                                'messages': [new_message]
                                })

        return call['text'], call['list_pictures']
