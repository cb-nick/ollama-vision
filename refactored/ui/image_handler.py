import streamlit as st
import base64
import time
from typing import Tuple, Optional

class MockFile:
    """Mock file object for webcam captures."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
    
    def read(self):
        return self._data

class ImageHandler:
    """Handles image input from various sources."""
    
    def __init__(self):
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 0
        if "show_camera" not in st.session_state:
            st.session_state.show_camera = False
    
    def handle_image_input(self) -> Tuple[Optional[object], Optional[bytes], Optional[str]]:
        """Handle image input from webcam or file upload."""
        # Camera input
        self._handle_camera_input()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload an image of the lost item",
            type=["jpg", "jpeg", "png"],
            key=st.session_state["file_uploader_key"],
        )
        
        # Clear webcam image if file is uploaded
        if uploaded_file is not None and "webcam_image" in st.session_state:
            del st.session_state["webcam_image"]
        
        # Return appropriate image source
        if "webcam_image" in st.session_state:
            return self._get_webcam_image()
        elif uploaded_file:
            return uploaded_file, uploaded_file.read(), "upload"
        
        return None, None, None
    
    def _handle_camera_input(self):
        """Handle camera input and capture."""
        if st.button("📷 Open Camera"):
            st.session_state.show_camera = not st.session_state.show_camera
            if "webcam_image" in st.session_state:
                del st.session_state["webcam_image"]
        
        if st.session_state.show_camera:
            webcam_file = st.camera_input("Capture an image of the lost item")
            if webcam_file:
                st.session_state.webcam_image = webcam_file.getvalue()
                st.session_state.webcam_filename = f"webcam_capture_{int(time.time())}.jpg"
                st.session_state.show_camera = False
                self._reset_uploader()
                st.rerun()
    
    def _get_webcam_image(self) -> Tuple[MockFile, bytes, str]:
        """Get webcam image data."""
        image_bytes = st.session_state.webcam_image
        image_file = MockFile(st.session_state.webcam_filename, image_bytes)
        return image_file, image_bytes, "webcam"
    
    def _reset_uploader(self):
        """Reset the file uploader widget."""
        st.session_state["file_uploader_key"] += 1
    
    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")
