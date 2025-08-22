import streamlit as st
import numpy as np
from ui.image_handler import ImageHandler
from typing import Optional

class UIComponents:
    """Handles UI component rendering."""
    
    def render_header(self):
        """Render the application header."""
        st.image("logo.png", width=250)
        st.title("Chargerback® AI — Vision Demo")
        st.text("Upload an image, and our AI-powered vision model will instantly analyze it, returning a detailed JSON description of the item(s) in the frame. This is a demonstration tool to showcase our capabilities.")
        st.info("Tip: For best results, use clear, well-lit photos with the item centered and in focus. Avoid cluttered backgrounds and ensure the entire item is visible.")
    
    def display_results(self, response: dict, response_time: float):
        """Display analysis results."""
        st.success(f"Response received in {response_time} seconds.")
        st.subheader("Results:")
        
        if "items" in response:
            for item in response["items"]:
                self._display_item_result(item)

            self._show_next_button(response)

        if "items" not in response or not response["items"]:
            st.info("No results found, please try another photo")
    
    def _display_item_result(self, item: dict):
        """Display results for a single item."""
        col1, col2 = st.columns([0.4, 0.6])
        
        # Display Chargerback type and product code
        cb_type = item.get("cb_type", "")
        product_code = item.get("product_code", "")
        
        cb_type_formatted = self._format_cb_type(cb_type)
        
        col1.markdown(f"##### Chargerback Type: :green[{cb_type_formatted or 'Unknown'}]")
        col1.markdown(f"##### Product Code: :green[{product_code or 'Unknown'}]")
        
        # Display CB attributes
        self._display_cb_attributes(col1, item)
        
        # Display other attributes
        with col2:
            item_filtered = self._filter_item_attributes(item)
            formatted_item = self._format_as_bullets(item_filtered)
            st.markdown(f"```\n{formatted_item}\n```")
        
        st.divider()

    def _show_next_button(self, response: dict):
        """Show the next button."""
        _, col2 = st.columns([0.9, 0.1])
        with col2:
            if st.button("Next"):
                print(f"Next button clicked. Response to pass along:\n{response}", flush=True)
                st.session_state["file_uploader_key"] += 1
                st.rerun()

    def _format_cb_type(self, cb_type: str) -> str:
        """Format CB type for display."""
        if not cb_type:
            return ""
        if cb_type.lower().startswith("ip"):
            return cb_type.replace("p", "P", 1)
        return cb_type[:1].upper() + cb_type[1:]
    
    def _display_cb_attributes(self, column, item: dict):
        """Display CB-specific attributes."""
        cb_attributes = {"brand", "amount", "currency_type", "color", "material", "case_color"}
        
        for attr in cb_attributes:
            if attr not in item:
                continue
                
            value = item[attr]
            if value == 0 or value is None or value == "":
                continue
            
            if isinstance(value, list):
                value = ", ".join(value)
            
            if not isinstance(value, (int, float, np.number)):
                value = value[0].upper() + value[1:] if value else "Unknown"
            
            column.markdown(f"##### {attr.replace('_', ' ').title()}: :green[{value}]")
    
    def _filter_item_attributes(self, item: dict) -> dict:
        """Filter out CB-specific attributes from item."""
        excluded = {"cb_type", "product_code", "type", "brand", "amount", 
                   "currency_type", "color", "material", "case_color"}
        return {k: v for k, v in item.items() if k not in excluded}
    
    def _format_as_bullets(self, data, indent: int = 0) -> str:
        """Format data as bulleted string."""
        formatted = ""
        prefix = "" if indent == 0 else "  " * indent + "- "
        
        if isinstance(data, dict):
            for key, value in data.items():
                formatted += f"{prefix}{str(key).replace('_', ' ').title()}:\n"
                formatted += self._format_as_bullets(value, indent + 1)
        elif isinstance(data, list):
            for item in data:
                formatted += self._format_as_bullets(item, indent)
        else:
            data_str = str(data).replace("_", " ")
            if data_str.lower().startswith("ip"):
                data_str = data_str.replace("p", "P")
            else:
                data_str = data_str[0].upper() + data_str[1:] if data_str else ""
            formatted += f"{prefix}{data_str}\n"
        
        return formatted
    
    def handle_feedback(self) -> Optional[str]:
        """Handle user feedback input."""
        sentiment_mapping = ["negative", "positive"]
        feedback_selected = st.feedback("thumbs", key="feedback_widget")
        
        if feedback_selected is not None:
            return sentiment_mapping[feedback_selected]
        return None
