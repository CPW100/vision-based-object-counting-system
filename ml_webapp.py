import streamlit as st

# Custom imports
import spage_one
import spage_two
import spage_three
import spage_four
import spage_five
import spage_six
import spage_seven
from smultipage import MultiPage

# D:\Program_Files\VisualStudio2019\streamlit_project\ml_webapp\ml_webapp > streamlit run ml_webapp.py

# Initializing Session State
session_state_keys = {"set_page_config_status": False}
for key, value in session_state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = value
if st.session_state["set_page_config_status"]==False:
    st.set_page_config(page_title = "Obeject Counting using ML", page_icon="Ã°Å¸Â§Å ", layout = "wide")
    st.session_state["set_page_config_status"] = True


# Initialize the multipage class
app = MultiPage()

# Title of the main pageD:
# Streamlit Interface
st.title("Obeject Counting using ML")
st.write(""" 
### This is a platform for you to train your own ML model to count for custom object which utilizes Tensorflow Object Detection API.""")

# Add all your applications (pages) here

app.add_page("Upload Data", spage_one.app)
app.add_page("Upload Data (Mask-RCNN)", spage_seven.app)
app.add_page("Configure Model Pipeline", spage_two.app)
app.add_page("User File Manager", spage_three.app)
app.add_page("Train My Model", spage_four.app)
app.add_page("Test My Model", spage_five.app)
app.add_page("Visualize Model Performance", spage_six.app)

# The main app
app.run()


































