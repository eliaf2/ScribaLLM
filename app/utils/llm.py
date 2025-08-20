from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from typing import TypedDict, Literal, List, Dict, Any
from pydantic import BaseModel, Field, SecretStr
from langgraph.graph import StateGraph, END, MessagesState
import base64
from PIL import Image
from io import BytesIO
import os
import streamlit as st
import logging
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition


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
    
    temperature = 0.2

    def __init__(self, openai_api_key: SecretStr, openai_llm_model: str, gemini_api_key: SecretStr, gemini_llm_model: str, context: str = ''):
        self.openai_api_key = openai_api_key
        self.openai_llm_model = openai_llm_model
        self.gemini_api_key = gemini_api_key
        self.gemini_llm_model = gemini_llm_model
        self.context = context if context else "converting written text to digital format"

        if not self.openai_api_key and not self.gemini_api_key:
            raise ValueError(
                "No API key provided. Please set either OpenAI or Gemini API key.")
        elif self.gemini_api_key:
            self.llm = init_chat_model(model=self.gemini_llm_model, model_provider="google_genai",
                                       api_key=self.gemini_api_key, temperature=self.temperature)
            logging.info(
                f"OCR_LLM initialized successfully with model: {self.gemini_llm_model}")
        elif self.openai_api_key:
            self.llm = init_chat_model(model=self.openai_llm_model, model_provider="openai",
                                       api_key=self.openai_api_key, temperature=self.temperature)
            logging.info(
                f"OCR_LLM initialized successfully with model: {self.openai_llm_model}")

        self.classifier_llm = self.llm.with_structured_output(
            ImageClassificationStructure)

        self.graph = StateGraph(AgentState)

        self.graph.add_node("classification", self._llm_classifier)
        self.graph.set_entry_point("classification")
        self.graph.add_node("ocr", self._ocr_function)

        self.graph.add_edge("classification", "ocr")
        self.graph.add_edge("ocr", END)

        self.app = self.graph.compile()

    # ======== Methods for the graph ========
    @staticmethod
    def _compress_image(
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

    def _llm_classifier(self, state: AgentState) -> AgentState:
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
        * Charts/graphs with text or formulas/equation labels
        * Diagrams with text labels or annotations
        * Quantum Circuits with labels or annotations
        * Artwork, drawings, or illustrations without text
        * Screenshots of documents

        Focus on whether there is meaningful text or textual information present in the image.

        Additionally, provide a brief description of what you see in the image using maximum 10 words considering the following context: {state['context']}"""

        for image_path in os.listdir(state['pictures_folder']):
            image_full_path = os.path.join(
                state['pictures_folder'], image_path)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                with BytesIO() as buffer:
                    self._compress_image(image_full_path).save(
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
                    logging.info(
                        f"Image: {image_path} classified as {response.classification}. It talks about {response.description}")      # type: ignore

                    if response.classification == "picture":    # type: ignore
                        state['list_pictures'].append(ImageOut(
                            path=image_full_path, description=response.description))    # type: ignore

        return state

    @staticmethod
    def _create_prompt_with_images(state: AgentState) -> str:
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

    def _ocr_function(self, state: AgentState) -> dict:
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
        system_prompt = self._create_prompt_with_images(state)
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

    def improve_ocr_result(self, ocr_results: list[str], context: str) -> str:

        ocr_improvement_prompt = f"""You are tasked with improving OCR-converted text while preserving its original text, structure and formatting. Please follow these guidelines:

            1. **Correct OCR errors**: Fix misread characters, words, and spacing issues
            2. **Preserve markdown formatting**: Maintain all original markdown syntax including lists, emphasis, code blocks, tables, etc.
            3. **Fix headers**: Using the context of the document, insert appropriate headers where necessary.
            4. **Fix LaTeX formulas**: Correct formatting errors in mathematical expressions while preserving their mathematical meaning
            5. **Preserve image links**: Do NOT modify any URLs or links that reference images

            ## Specific Rules:

            ### Text Correction:
            - Fix obvious character recognition errors (e.g., "rn" misread as "m", "cl" as "d", "0" as "O")
            - Correct spacing issues and word boundaries
            - Fix punctuation errors
            - Maintain the original language and writing style

            ### Markdown Formatting:
            - Keep all headers (#, ##, ###, etc.) intact
            - Preserve bullet points, numbered lists, and indentation
            - Maintain emphasis markers (**, *, ~~, etc.)
            - Keep code blocks (```) and inline code (``) formatting
            - Maintain block quotes (>) and other markdown elements
            - Maintain blockquotes (>) and other markdown elements

            ### LaTeX Mathematical Expressions:
            - Fix syntax errors in LaTeX formulas (missing brackets, incorrect commands, etc.)
            - Correct common OCR mistakes in mathematical notation:
            - Greek letters
            - Mathematical operators
            - Fraction notation
            - Ensure proper bracket matching and command structure

            ### Image Links:
            - **NEVER modify image URLs or file paths**
            - Keep all image reference syntax intact: `![alt text](image_url)` or `<img src="..."/>`
            - Preserve any image-related markdown or HTML tags

            ### Quality Control:
            - Read the text in context to ensure corrections make logical sense
            - When uncertain about a correction, err on the side of minimal changes
            - Maintain consistency in terminology and formatting throughout the document

            ## Output Format:
            Return only the corrected text with no additional commentary, explanations, or metadata. The output should be ready to use as-is.

            ---

            **The text talks about:** {context}"""

        ocr_correction_prompt = f"""You are an OCR correction specialist. Your task is to fix OCR conversion errors in the following markdown text and rewrite it in a more readable format.

        STRICT RULES:
        - Do NOT remove any content or information from the original text
        - Do NOT summarize or condense any sections
        - You may ONLY ADD content (formatting, punctuation, line breaks, headers, etc.)
        - Fix obvious OCR errors (character misrecognition, spacing issues, etc.)
        - Improve formatting and structure for better readability
        - Add proper markdown formatting where appropriate
        - Maintain the original meaning and all details

        The original OCR text is provided by the user.

        Please provide the corrected and improved version while preserving ALL original content."""

        temperature = 0.3

        if not self.openai_api_key and not self.gemini_api_key:
            raise ValueError(
                "No API key provided. Please set either OpenAI or Gemini API key.")
        elif self.gemini_api_key:
            optimizer_llm = init_chat_model(model=self.gemini_llm_model, model_provider="google_genai",
                                            api_key=self.gemini_api_key, temperature=temperature)
        elif self.openai_api_key:
            optimizer_llm = init_chat_model(model=self.openai_llm_model, model_provider="openai",
                                            api_key=self.openai_api_key, temperature=temperature)

        text = ''
        for page in ocr_results:
            text += ''.join(page) + '\n\n'

        new_message = HumanMessage(content=text)
        response = optimizer_llm.invoke(
            [SystemMessage(content=ocr_correction_prompt)] + [new_message])

        return response.content if response else text    # type: ignore

# =======================================


class ChromaConfig(BaseModel):
    chroma_path: str = "chroma"
    data_path: str = "data"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "rag_documents"
    file_pattern: str = "*.md"
    embedding_model: str = "text-embedding-3-small"
    batch_size: int = 100
    force_rebuild: bool = False


class ChromaVectorStore:
    def __init__(self, config: ChromaConfig, openai_api_key: SecretStr):
        self.config = config
        self.embeddings = None
        self.db = None
        self.metadata_file = Path(config.chroma_path) / \
            "document_metadata.json"
        self.openai_api_key = openai_api_key

    def _initialize_embeddings(self) -> None:
        '''Initialize OpenAI embeddings with error handling.
        '''
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.openai_api_key,
                max_retries=3,
                request_timeout=60  # type: ignore
            )
            # Test the embedding connection
            test_embedding = self.embeddings.embed_query("test")
            logging.info(
                f"Embeddings initialized successfully with model: {self.config.embedding_model}")
        except Exception as e:
            logging.error(f"Failed to initialize embeddings: {e}")
            st.error(
                "Failed to initialize embeddings. Please check your API key and model configuration.")
            raise

    def _load_existing_metadata(self) -> Dict[str, Any]:
        '''Load existing document metadata for incremental updates.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the document metadata.
        '''
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load metadata file: {e}")
        return {"documents": {}, "last_updated": None}

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        '''Save document metadata for future incremental updates.

        Parameters
        ----------
        metadata : Dict[str, Any]
            A dictionary containing the document metadata.
        '''
        try:
            os.makedirs(Path(self.config.chroma_path), exist_ok=True)
            metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")

    def _calculate_source_hash(self, documents_from_source: List[Document]) -> str:
        '''Calculate hash of all documents from a single source file.

        Parameters
        ----------
        documents_from_source : List[Document]
            A list of Document objects representing the source documents.

        Returns
        -------
        str
            The MD5 hash of the combined content and metadata.
        '''
        combined_content = "".join(
            [doc.page_content for doc in documents_from_source])
        basic_metadata = {k: v for k, v in documents_from_source[0].metadata.items()
                          if k in ['source', 'filename'] and not k.startswith('chunk')}
        combined_metadata = str(sorted(basic_metadata.items()))
        return hashlib.md5((combined_content + combined_metadata).encode('utf-8')).hexdigest()

    def load_documents(self) -> List[Document]:
        '''Load documents from the specified directory with comprehensive error handling.

        Returns
        -------
        List[Document]
            A list of Document objects representing the loaded documents.

        Raises
        ------
        ValueError
            If no documents are found or if an error occurs during loading.
        '''
        try:
            logging.info(
                f"Loading documents from: {self.config.data_path}")
            loader = DirectoryLoader(
                self.config.data_path,
                glob=self.config.file_pattern,
                show_progress=True,
                use_multithreading=True
            )
            documents = loader.load()

            if not documents:
                raise ValueError(
                    f"No documents found in {self.config.data_path} with pattern {self.config.file_pattern}")

            logging.info(
                f"Successfully loaded {len(documents)} documents for embedding.")
            return documents

        except Exception as e:
            logging.error(f"Failed to load documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        '''Split documents into chunks with optimized parameters.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects to be split into chunks.

        Returns
        -------
        List[Document]
            A list of Document objects representing the split chunks.
        '''
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                add_start_index=True,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = text_splitter.split_documents(documents)

            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content),
                    "total_chunks": len(chunks)
                })

            logging.info(
                f"Split {len(documents)} documents into {len(chunks)} chunks")

            if chunks:
                sample_chunk = chunks[min(10, len(chunks) - 1)]
                logging.debug(
                    f"Sample chunk content: {sample_chunk.page_content[:200]}...")
                logging.debug(
                    f"Sample chunk metadata: {sample_chunk.metadata}")

            return chunks

        except Exception as e:
            logging.error(f"Failed to split documents: {e}")
            raise

    def _needs_update(self, documents: List[Document]) -> tuple[bool, List[Document]]:
        '''Determine if the database needs to be updated based on the provided documents.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects to check for updates.

        Returns
        -------
        tuple[bool, List[Document]]
            A tuple containing a boolean indicating if an update is needed,
            and a list of Document objects that are new or changed.
        '''
        if self.config.force_rebuild:
            logging.info(
                "Force rebuild requested, processing all documents.")
            return True, documents

        existing_metadata = self._load_existing_metadata()
        new_documents = []

        # Group documents by source file to avoid processing chunks individually
        documents_by_source = {}
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in documents_by_source:
                documents_by_source[source] = []
            documents_by_source[source].append(doc)

        # Check each source file for changes
        for source, docs in documents_by_source.items():
            combined_content = "".join([doc.page_content for doc in docs])
            combined_metadata = str(sorted(docs[0].metadata.items()))
            doc_hash = hashlib.md5(
                (combined_content + combined_metadata).encode('utf-8')).hexdigest()

            if source not in existing_metadata["documents"] or \
               existing_metadata["documents"][source] != doc_hash:
                new_documents.extend(docs)
                logging.debug(f"Source file changed: {source}")
            else:
                logging.debug(f"Source file unchanged: {source}")

        needs_update = len(new_documents) > 0
        if needs_update:
            changed_sources = set(doc.metadata.get(
                'source', 'unknown') for doc in new_documents)
            logging.info(
                f"Found changes in {len(changed_sources)} source files, processing {len(new_documents)} documents.")
        else:
            logging.info("No document changes detected, skipping update.")

        return needs_update, new_documents

    def _process_chunks_in_batches(self, chunks: List[Document]) -> None:
        '''Process chunks in batches for better memory management.

        Parameters
        ----------
        chunks : List[Document]
            A list of Document objects to be processed in batches.
        '''
        total_batches = (len(chunks) + self.config.batch_size -
                         1) // self.config.batch_size

        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batch_num = (i // self.config.batch_size) + 1

            try:
                logging.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")

                if self.db is None:
                    self.db = Chroma.from_documents(
                        batch,
                        self.embeddings,
                        persist_directory=self.config.chroma_path,
                        collection_name=self.config.collection_name
                    )
                else:
                    self.db.add_documents(batch)

            except Exception as e:
                logging.error(f"Failed to process batch {batch_num}: {e}")
                raise

    def save_to_chroma(self, chunks: List[Document]) -> None:
        '''Save chunks to Chroma with incremental update support and batch processing.

        Parameters
        ----------
        chunks : List[Document]
            A list of Document objects to be saved to Chroma.
        '''
        try:
            if not self.embeddings:
                self._initialize_embeddings()

            documents = self.load_documents()
            needs_update, documents_to_process = self._needs_update(documents)

            if not needs_update and not self.config.force_rebuild:
                logging.info("Vector database is up to date.")
                return

            if self.config.force_rebuild and os.path.exists(self.config.chroma_path):
                logging.info("Removing existing database for rebuild.")
                shutil.rmtree(self.config.chroma_path)

            if documents_to_process:
                chunks_to_process = self.split_documents(documents_to_process)

                logging.info(
                    f"Saving {len(chunks_to_process)} chunks from {len(documents_to_process)} changed documents to Chroma.")

                if not self.config.force_rebuild and os.path.exists(self.config.chroma_path):
                    self.db = Chroma(
                        persist_directory=self.config.chroma_path,
                        embedding_function=self.embeddings,
                        collection_name=self.config.collection_name
                    )

                self._process_chunks_in_batches(chunks_to_process)

                metadata = self._load_existing_metadata()

                documents_by_source = {}
                for doc in documents:
                    source = doc.metadata.get('source', 'unknown')
                    if source not in documents_by_source:
                        documents_by_source[source] = []
                    documents_by_source[source].append(doc)

                for source, docs in documents_by_source.items():
                    doc_hash = self._calculate_source_hash(docs)
                    metadata["documents"][source] = doc_hash

                self._save_metadata(metadata)

                logging.info(
                    f"Successfully saved {len(chunks_to_process)} chunks to {self.config.chroma_path}.")

        except Exception as e:
            logging.error(f"Failed to save to Chroma: {e}")
            raise

    def get_database_stats(self) -> Dict[str, Any]:
        '''Get statistics about the current database.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing database statistics.
        '''
        try:
            if not os.path.exists(self.config.chroma_path):
                return {"status": "Database not found"}

            if not self.embeddings:
                self._initialize_embeddings()

            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name
            )

            collection = db._collection
            stats = {
                "status": "Active",
                "total_documents": collection.count(),
                "database_path": self.config.chroma_path,
                "collection_name": self.config.collection_name,
                "embedding_model": self.config.embedding_model
            }

            if self.metadata_file.exists():
                metadata = self._load_existing_metadata()
                stats["last_updated"] = metadata.get("last_updated")
                stats["tracked_documents"] = len(metadata.get("documents", {}))

            return stats

        except Exception as e:
            logging.error(f"Failed to get database stats: {e}")
            return {"status": "Error", "error": str(e)}

    def generate_data_store(self) -> None:
        '''Generate the vector data store from the loaded documents.
        '''
        try:
            logging.info("Starting vector database generation")

            documents = self.load_documents()

            needs_update, _ = self._needs_update(documents)

            if not needs_update and not self.config.force_rebuild:
                logging.info(
                    "No changes detected, database is already up to date.")
                stats = self.get_database_stats()
                logging.info(f"Current database stats: {stats}")
                return

            chunks = self.split_documents(documents)

            self.save_to_chroma(chunks)

            # Display final stats
            stats = self.get_database_stats()
            logging.info(f"Database generation completed. Stats: {stats}")

        except Exception as e:
            logging.error(f"Vector database generation failed: {e}")
            raise

    def list_documents(self) -> pd.DataFrame:
        '''List all documents in the database and return as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information about all documents in the database.
            Columns include: document_id, source, filename, chunk_id, chunk_size, 
            total_chunks, content_preview, and any other metadata.
        '''
        try:
            if not os.path.exists(self.config.chroma_path):
                logging.warning("Database not found")
                return pd.DataFrame()

            if not self.embeddings:
                self._initialize_embeddings()

            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name
            )

            # Get all documents from the collection
            collection = db._collection
            results = collection.get()

            if not results['documents']:
                logging.info("No documents found in the database")
                return pd.DataFrame()

            # Prepare data for DataFrame
            data = []
            for i, (doc_id, document, metadata) in enumerate(zip(   # type: ignore
                results['ids'],
                results['documents'],
                results['metadatas']    # type: ignore
            )):
                row = {
                    'document_id': doc_id,
                    'content': document,
                    'content_preview': document[:200] + '...' if len(document) > 200 else document,
                    'content_length': len(document),
                }

                # Add all metadata fields
                if metadata:
                    row.update(metadata)

                data.append(row)

            df = pd.DataFrame(data)

            # Reorder columns
            priority_columns = ['document_id', 'source', 'filename', 'chunk_id',
                                'chunk_size', 'total_chunks', 'content_preview', 'content_length']

            # Get existing columns in priority order, then add remaining columns
            existing_priority_cols = [
                col for col in priority_columns if col in df.columns]
            remaining_cols = [
                col for col in df.columns if col not in priority_columns]

            if existing_priority_cols:
                df = df[existing_priority_cols + remaining_cols]

            logging.info(f"Retrieved {len(df)} documents from the database")
            return df

        except Exception as e:
            logging.error(f"Failed to list documents: {e}")
            return pd.DataFrame()

    def remove_documents_by_source(self, source_path: str) -> Dict[str, Any]:
        '''Remove all embeddings for documents from a specific source file.

        Parameters
        ----------
        source_path : str
            The source path/filename to remove from the database.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing information about the removal operation,
            including the number of documents removed and operation status.
        '''
        try:
            if not os.path.exists(self.config.chroma_path):
                return {
                    "status": "error",
                    "message": "Database not found",
                    "documents_removed": 0
                }

            if not self.embeddings:
                self._initialize_embeddings()

            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name
            )

            collection = db._collection

            results = collection.get()

            if not results['documents']:
                return {
                    "status": "success",
                    "message": "No documents found in database",
                    "documents_removed": 0
                }

            # Find document IDs that match the source
            matching_ids = []
            for doc_id, metadata in zip(results['ids'], results['metadatas']):  # type: ignore
                if metadata and metadata.get('source') == source_path:
                    matching_ids.append(doc_id)

            if not matching_ids:
                return {
                    "status": "success",
                    "message": f"No documents found with source: {source_path}",
                    "documents_removed": 0,
                    "available_sources": list(set(
                        metadata.get('source', 'unknown')
                        for metadata in results['metadatas']    # type: ignore
                        if metadata
                    ))
                }

            collection.delete(ids=matching_ids)

            # Update metadata file to remove the source
            metadata = self._load_existing_metadata()
            if source_path in metadata.get("documents", {}):
                del metadata["documents"][source_path]
                self._save_metadata(metadata)
            

            logging.info(
                f"Removed {len(matching_ids)} documents from source: {source_path}")
            
            # Delete the source file from disk
            try:
                if os.path.exists(source_path):
                    os.remove(source_path)
                    logging.info(f"Deleted source file: {source_path}")
                else:
                    logging.warning(f"Source file not found: {source_path}")
            except Exception as e:
                logging.error(f"Failed to delete source file {source_path}: {e}")

            return {
                "status": "success",
                "message": f"Successfully removed documents from source: {source_path}",
                "documents_removed": len(matching_ids),
                "removed_document_ids": matching_ids
            }

        except Exception as e:
            error_msg = f"Failed to remove documents from source {source_path}: {e}"
            logging.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "documents_removed": 0
            }

    def get_source_statistics(self) -> pd.DataFrame:
        '''Get aggregated statistics for each source file in the database.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing statistics aggregated by source file.
            Columns include: source, filename, total_chunks, total_content_length,
            avg_chunk_size, min_chunk_size, max_chunk_size, first_chunk_id, last_chunk_id.
        '''
        try:
            if not os.path.exists(self.config.chroma_path):
                logging.warning("Database not found")
                return pd.DataFrame()
            
            if not self.embeddings:
                self._initialize_embeddings()
            
            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name
            )
            
            # Get all documents from the collection
            collection = db._collection
            results = collection.get()
            
            if not results['documents']:
                logging.info("No documents found in the database")
                return pd.DataFrame()
            
            # Group data by source
            source_data = {}
            
            for doc_id, document, metadata in zip(
                results['ids'], 
                results['documents'], 
                results['metadatas']    # type: ignore
            ):
                if not metadata or 'source' not in metadata:
                    continue
                    
                source = metadata['source']
                
                if source not in source_data:
                    source_data[source] = {
                        'documents': [],
                        'chunk_sizes': [],
                        'chunk_ids': [],
                        'metadata': metadata
                    }
                
                source_data[source]['documents'].append({
                    'id': doc_id,
                    'content': document,
                    'content_length': len(document),
                    'metadata': metadata
                })
                
                chunk_size = metadata.get('chunk_size', len(document))
                source_data[source]['chunk_sizes'].append(chunk_size)
                
                chunk_id = metadata.get('chunk_id', 0)
                source_data[source]['chunk_ids'].append(chunk_id)
            
            # Calculate statistics for each source
            stats_data = []
            
            for source, data in source_data.items():
                documents = data['documents']
                chunk_sizes = data['chunk_sizes']
                chunk_ids = data['chunk_ids']
                sample_metadata = data['metadata']
                
                # Extract filename from source path
                filename = os.path.basename(source) if source else 'unknown'
                
                # Calculate statistics
                total_chunks = len(documents)
                total_content_length = sum(doc['content_length'] for doc in documents)
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
                max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
                min_chunk_id = min(chunk_ids) if chunk_ids else 0
                max_chunk_id = max(chunk_ids) if chunk_ids else 0
                
                # Get total chunks from metadata (if available)
                total_chunks_in_source = sample_metadata.get('total_chunks', total_chunks)
                
                # Calculate additional metrics
                avg_content_length = total_content_length / total_chunks if total_chunks > 0 else 0
                
                # Check if this source appears to be complete (all chunks present)
                expected_chunk_ids = set(range(total_chunks_in_source)) if total_chunks_in_source else set()
                actual_chunk_ids = set(chunk_ids)
                is_complete = expected_chunk_ids.issubset(actual_chunk_ids) if expected_chunk_ids else True
                missing_chunks = len(expected_chunk_ids - actual_chunk_ids) if expected_chunk_ids else 0
                
                stats_row = {
                    'source': source,
                    'filename': filename,
                    'total_chunks': total_chunks,
                    'total_chunks_expected': total_chunks_in_source,
                    'missing_chunks': missing_chunks,
                    'is_complete': is_complete,
                    'total_content_length': total_content_length,
                    'avg_content_length': round(avg_content_length, 2),
                    'avg_chunk_size': round(avg_chunk_size, 2),
                    'min_chunk_size': min_chunk_size,
                    'max_chunk_size': max_chunk_size,
                    'chunk_id_range': f"{min_chunk_id}-{max_chunk_id}",
                    'first_chunk_id': min_chunk_id,
                    'last_chunk_id': max_chunk_id,
                }
                
                # Add any additional metadata that might be useful
                for key in ['filename', 'source']:
                    if key in sample_metadata and key not in stats_row:
                        stats_row[f'metadata_{key}'] = sample_metadata[key]
                
                stats_data.append(stats_row)
            
            # Create DataFrame
            df = pd.DataFrame(stats_data)
            
            # Sort by source name for consistent ordering
            if not df.empty:
                df = df.sort_values('source').reset_index(drop=True)
            
            logging.info(f"Generated statistics for {len(df)} sources")
            return df
            
        except Exception as e:
            logging.error(f"Failed to get source statistics: {e}")
            return pd.DataFrame()


# =======================================


class ChatbotMessagesState(MessagesState):
    human_msg_positions: list[int]


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class ChatbotLLM:
    
    temperature = 1.0

    def __init__(self, openai_api_key: SecretStr, openai_llm_model: str, chroma_config: ChromaConfig):
        self.openai_api_key = openai_api_key
        self.openai_llm_model = openai_llm_model
        self.chroma_config = chroma_config

        self.llm = init_chat_model(model=self.openai_llm_model, model_provider="openai",
                                   api_key=self.openai_api_key, temperature=self.temperature)
        self.grader_model = init_chat_model(model=self.openai_llm_model, model_provider="openai",
                                            api_key=self.openai_api_key, temperature=0.0)
        self.grader_model = self.grader_model.with_structured_output(
            GradeDocuments)

        self.vectorstore = Chroma(
            persist_directory=self.chroma_config.chroma_path,
            embedding_function=OpenAIEmbeddings(
                api_key=openai_api_key, model=self.chroma_config.embedding_model),
            collection_name=self.chroma_config.collection_name
        )
        self.retriever = self.vectorstore.as_retriever()
        retriever_description = f"""Use this tool to search the user’s private technical notes when a question may involve detailed, 
        domain-specific, or previously recorded information. The notes include high-level technical content in mathematics, 
        physics, and science (such as formulas, derivations, definitions, proofs, physical laws, and scientific explanations). 
        Always call this tool if the user requests specific technical details or asks about 
        concepts that may be more accurate, complete, or personalized if retrieved from their notes rather than general knowledge."""

        self.retriever_tool = create_retriever_tool(
            retriever=self.retriever,
            name="NotesRetriever",
            description=retriever_description
        )

        self.graph = StateGraph(ChatbotMessagesState)

        self.graph.add_node("generate_query_or_respond",
                            self._generate_query_or_respond)
        self.graph.add_node("retrieve", ToolNode([self.retriever_tool]))
        self.graph.add_node("rewrite_question", self._rewrite_question)
        self.graph.add_node("generate_answer", self._generate_answer)

        self.graph.set_entry_point("generate_query_or_respond")

        self.graph.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        self.graph.add_conditional_edges(
            "retrieve",
            self._grade_documents,
        )
        self.graph.add_edge("generate_answer", END)
        self.graph.add_edge("rewrite_question", "generate_query_or_respond")

        self.app = self.graph.compile()

    def _rewrite_question(self, state: ChatbotMessagesState) -> Dict[str, Any]:
        '''Rewrite the original user question.

        Parameters
        ----------
        state : ChatbotMessagesState
            The current state of the chatbot messages.

        Returns
        -------
        Dict[str, Any]
            The updated state with the rewritten question appended to messages.
        '''
        system_prompt = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "- {question}"
            "\n ------- \n"
            "Here there are the previous messages of the user:"
            "\n ------- \n"
            "- {previous_messages}"
            "\n ------- \n"
            "Formulate an improved question:"
        )
        messages = state["messages"]
        question = messages[1].content
        previous_messages_positions = state["human_msg_positions"]
        previous_messages = "\n- ".join([str(messages[i].content)
                                        for i in previous_messages_positions])
        prompt = system_prompt.format(
            question=question, previous_messages=previous_messages)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    def _generate_query_or_respond(self, state: ChatbotMessagesState) -> Dict[str, Any]:
        '''Call the model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply respond to the user.

        Parameters
        ----------
        state : ChatbotMessagesState
            The current state of the chatbot messages.

        Returns
        -------
        Dict[str, Any]
            The updated state with the generated response appended to messages.
        '''
        response = (
            self.llm
            .bind_tools([self.retriever_tool]).invoke(state["messages"])
        )
        return {"messages": [response]}

    def _grade_documents(self, state: ChatbotMessagesState) -> Literal["generate_answer", "rewrite_question"]:
        '''Determine whether the retrieved documents are relevant to the question.

        Parameters
        ----------
        state : ChatbotMessagesState
            The current state of the chatbot messages.

        Returns
        -------
        Literal["generate_answer", "rewrite_question"]
            The next action to take based on the relevance of the documents.
        '''
        system_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )

        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = system_prompt.format(question=question, context=context)
        response = (
            self.grader_model.invoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score   # type: ignore

        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    def _generate_answer(self, state: ChatbotMessagesState) -> Dict[str, Any]:
        '''Generate an answer based on the current state.

        Parameters
        ----------
        state : ChatbotMessagesState
            The current state of the chatbot messages.

        Returns
        -------
        Dict[str, Any]
            The updated state with the generated answer appended to messages.
        '''
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \n"
            "Context: {context}"
        )

        question = state["messages"][1].content
        context = state["messages"][-1].content
        prompt = system_prompt.format(question=question, context=context)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    def __call__(self, messages: list, human_msgs_position: list[int]) -> dict[str, Any] | Any:
        response = self.app.invoke(
            {"messages": messages, "human_msg_positions": human_msgs_position})
        return response if response else {"messages": []}
