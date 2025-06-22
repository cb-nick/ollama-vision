import os
import sys
import time
import json
from typing import Optional

import numpy as np
import streamlit as st
import supervision as sv
import torch
from api.grounding_dino_model import GroundingDINOModel
from dotenv import find_dotenv, load_dotenv
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import HumanMessage
from langchain.tools import BaseTool
from loguru import logger
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

_ = load_dotenv(find_dotenv())

# logger.remove()
# logger.add(sys.stderr, level="INFO")

logger.remove()
logger.add("image_auto_annotation.log", rotation="1 MB", level="INFO", backtrace=True, diagnose=True)

# st.set_page_config(layout="centered")
st.set_page_config(layout="wide")

class ImageDescriber:
    def __init__(self, model_name: str, device: str) -> None:
        self._device = device
        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    def __call__(self, image_path: str) -> str:
        image_obj = Image.open(image_path).convert("RGB")
        inputs = self._processor(image_obj, return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs)
        return self._processor.decode(output[0], skip_special_tokens=True)

class ImageDescriberTool(BaseTool):
    name: str = "Describe image tool"
    description: str = "Use this tool to describe found objects in an image"
    image_describer: Optional[ImageDescriber] = None

    def setup(self, image_describer: ImageDescriber) -> BaseTool:
        self.image_describer = image_describer
        return self

    def _run(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return self.image_describer(image_path)

    def _arun(self, query: str):
        raise NotImplementedError

class PromptGeneratorTool(BaseTool):
    name: str = "Image object detection prompt generator tool"
    description: str = "Use this tool to generate prompt based on the description of the image"
    llm: Optional[BaseLanguageModel] = None

    def setup(self, llm: BaseLanguageModel) -> BaseTool:
        self.llm = llm
        return self

    def _run(self, image_desc: str) -> str:
        input_msg = [HumanMessage(content=f"List only the object names in this sentence, separated by commas: {image_desc}")]
        gen_prompt = self.llm.invoke(input_msg)
        return gen_prompt.content if hasattr(gen_prompt, 'content') else str(gen_prompt)

    def _arun(self, query: str):
        raise NotImplementedError

class ObjectDetectionTool(BaseTool):
    name: str = "Object detection on image tool"
    description: str = "Detect objects in an image with a text prompt"
    groundingDINO_model: Optional[GroundingDINOModel] = None
    output_quality: int = 70

    def setup(self, groundingDINO_model: GroundingDINOModel, output_quality=70) -> BaseTool:
        self.groundingDINO_model = groundingDINO_model
        self.output_quality = output_quality
        return self

    def _run(self, image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        detections, labels = self.groundingDINO_model(False, image=image_np, caption=prompt)

        if len(detections.xyxy) > 0:
            detections.labels = labels
            annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            image_np = annotator.annotate(scene=image_np, detections=detections)

        output_dir = "output/"
        os.makedirs(output_dir, exist_ok=True)
        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_img_path = os.path.join(output_dir, f"{now_time}.png")
        Image.fromarray(image_np).save(output_img_path, format="PNG", optimize=True, quality=self.output_quality)
        return output_img_path

    def _arun(self, query: str):
        raise NotImplementedError

class App:
    def __init__(self, device) -> None:
        self.groundingDINO_model = GroundingDINOModel.create_instance(
            device=device,
            groundingDINO_type=st.session_state.get("groundingDINO_model", "swint_ogc")
        ).setup(
            box_threshold=st.session_state.get("box_threshold", 0.35),
            text_threshold=st.session_state.get("text_threshold", 0.25)
        )

        self.image_describer = ImageDescriber(
            st.session_state.get("blip-image-captioning", "Salesforce/blip-image-captioning-base"),
            device
        )

        self.output_quality = st.session_state.get("output_quality", 70)

        if "agent" not in st.session_state:
            llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
            self._agent = initialize_agent(
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                tools=[
                    ImageDescriberTool().setup(self.image_describer),
                    ObjectDetectionTool().setup(self.groundingDINO_model, self.output_quality),
                    PromptGeneratorTool().setup(llm),
                ],
                llm=llm,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            )
            st.session_state["agent"] = self._agent
        else:
            self._agent = st.session_state["agent"]

    def _upload_image(self):
        uploaded_image = st.file_uploader("Upload an image")
        if uploaded_image:
            tmp_dir = "tmp/"
            os.makedirs(tmp_dir, exist_ok=True)
            temp_file_path = os.path.join(tmp_dir, uploaded_image.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_image.getvalue())
            self._image_agents_handler(temp_file_path)

    def _image_agents_handler(self, image_path: str):
        try:
            st.subheader("Image Auto Annotation")
            st.image(image_path, caption="Uploaded Image", width=200)

            start_time = time.time()
            agent_response = self._agent.run({"input": image_path})
            elapsed = time.time() - start_time

            st.subheader("Structured Item Description")
            try:
                if isinstance(agent_response, dict):
                    st.json(agent_response)
                else:
                    parsed_data = json.loads(str(agent_response))
                    st.json(parsed_data)
            except (json.JSONDecodeError, TypeError):
                st.code(str(agent_response))

            st.caption(f"⏱️ Processed in {elapsed:.2f} seconds")

        except Exception as e:
            logger.error(e)
            st.error(f"❌ Error analyzing image: {str(e)}")

    def run(self):
        st.title("Image Auto Annotation")
        self._upload_image()

        with st.sidebar:
            st.radio("Captioning Model", [
                "Salesforce/blip-image-captioning-base",
                "Salesforce/blip-image-captioning-large"
            ], key="blip-image-captioning")
            st.slider("Box threshold", 0.0, 1.0, 0.35, key="box_threshold")
            st.slider("Text threshold", 0.0, 1.0, 0.25, key="text_threshold")
            st.radio("GroundingDINO Model", ["swint_ogc", "swinb_cogcoor"], key="groundingDINO_model")
            st.slider("Image quality", 0, 100, 70, key="output_quality")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()