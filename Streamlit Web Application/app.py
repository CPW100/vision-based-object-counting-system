import streamlit as st
import os

# Custom import
import colab_test
# import spage_two
# import spage_three
# import spage_four
# import spage_five
# import spage_six
# import spage_seven
from smultipage import MultiPage

# Initializing Session State
session_state_keys = {"set_page_config_status": False}
for key, value in session_state_keys.items():
  if key not in st.session_state:
    st.session_state[key] = value
if st.session_state["set_page_config_status"]==False:
  st.set_page_config(page_title = "Obeject Counting using ML", page_icon="🧊", layout = "wide")
  st.session_state["set_page_config_status"] = True

# Initialize the multipage class
app = MultiPage()

def page_one():
 st.selectbox('CHoose an option', ['option 1', 'option 2', 'option 3'])

# Title of the main page:
# Streamlit Interface
st.title("Obeject Counting using ML")
st.write(""" 
### This is a platform for you to train your own ML model to count for custom object which utilizes Tensorflow Object Detection API.""")
# Add all your applications (pages) here
app.add_page("Upload Data", page_one)

app.run

# def main():



# if __name__ == '__main__':
# 	main()
