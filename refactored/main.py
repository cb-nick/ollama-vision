import streamlit as st
from config import AppConfig
from services.data_loader import DataLoader
from services.database import DatabaseService
from services.vision_service import VisionService
from services.semantic_search import SemanticSearchService
from ui.components import UIComponents
from ui.image_handler import ImageHandler
from utils.logger import setup_logging
import time

# Setup logging
setup_logging()

class StreamlitApp:
    def __init__(self):
        self.config = AppConfig()
        self.db_service = DatabaseService()
        self.vision_service = VisionService(self.config)
        self.semantic_search = SemanticSearchService(self.config)
        self.ui_components = UIComponents()
        self.image_handler = ImageHandler()
        
        # Initialize data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize application data and cache it in session state."""
        if "data_loaded" not in st.session_state:
            with st.spinner("Loading application data..."):
                data_loader = DataLoader()
                data = data_loader.load_all_data()
                
                # Store in session state
                for key, value in data.items():
                    st.session_state[key] = value
                
                st.session_state.data_loaded = True
                st.session_state.semantic_search = self.semantic_search
    
    def _reset_analysis_state(self):
        """Reset analysis-related session state for new images."""
        analysis_keys = ['response', 'response_time', 'feedback', 'saved_types']
        for key in analysis_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    def _handle_new_image(self, image_id: str):
        """Handle processing of a new image."""
        if "last_image_id" not in st.session_state or st.session_state.last_image_id != image_id:
            self._reset_analysis_state()
            st.session_state.last_image_id = image_id
            st.session_state.request_id = self.db_service.generate_request_id()
    
    def _analyze_image(self, image_bytes: bytes) -> dict:
        """Analyze the uploaded image and return results."""
        try:
            start_time = time.time()
            result = self.vision_service.analyze_image(image_bytes)
            end_time = time.time()
            
            response_time = round(end_time - start_time, 2)
            
            # Enrich results with semantic search
            if result and "items" in result:
                for item in result["items"]:
                    cb_type, product_code = self.semantic_search.find_closest_match(
                        item.get("type", "unknown")
                    )
                    item["cb_type"] = cb_type
                    item["product_code"] = product_code
            
            return {
                "result": result,
                "response_time": response_time,
                "success": True
            }
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run(self):
        """Main application entry point."""
        # Setup page
        st.set_page_config(page_title="Chargerback Vision Demo", layout="centered")
        self.ui_components.render_header()
        
        # Handle image input
        image_file, image_bytes, image_source = self.image_handler.handle_image_input()
        
        if image_file and image_bytes:
            # Handle new image
            image_name = getattr(image_file, "name", "uploaded_image")
            image_id = f"{image_source}_{image_name}"
            self._handle_new_image(image_id)
            
            # Store image in session state
            st.session_state.image = self.image_handler.encode_image(image_bytes)
            
            # Display image
            st.image(image_bytes, caption=f"Image from {image_source}", use_container_width=True)
            
            # Initialize session state variables
            if "response" not in st.session_state:
                st.session_state.response = None
            if "response_time" not in st.session_state:
                st.session_state.response_time = None
            if "feedback" not in st.session_state:
                st.session_state.feedback = None
            
            # Analysis button
            if st.button("🔍 Analyze"):
                with st.spinner("⏳ Analyzing image..."):
                    analysis_result = self._analyze_image(image_bytes)
                
                if analysis_result["success"]:
                    st.session_state.response = analysis_result["result"]
                    st.session_state.response_time = analysis_result["response_time"]
                    st.session_state.feedback = None
                    
                    # Log the response
                    self.db_service.log_response(
                        st.session_state.request_id,
                        st.session_state.image,
                        st.session_state.response,
                        st.session_state.response_time
                    )
            
            # Display results
            if st.session_state.response:
                self.ui_components.display_results(
                    st.session_state.response,
                    st.session_state.response_time if st.session_state.response_time is not None else 0.0
                )
                
                # Handle feedback
                feedback = self.ui_components.handle_feedback()
                if feedback is not None:
                    st.session_state.feedback = feedback
                    self.db_service.log_feedback(st.session_state.request_id, feedback)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
