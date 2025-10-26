import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Missing Person Management Dashboard")

st.subheader("Add New Record")
with st.form("add_form"):
    name = st.text_input("Name")
    age = st.number_input("Age", 0, 120)
    contact = st.text_input("Contact")
    image = st.file_uploader("Upload Image", type=["jpg","png"])
    submitted = st.form_submit_button("Submit")

    if submitted and image:
        files = {"image": image}
        data = {"name": name, "age": age, "contact": contact}
        res = requests.post(f"{API_URL}/add_child/", data=data, files=files)
        st.success(res.json()["message"])

st.subheader("Database Records")
res = requests.get(f"{API_URL}/get_children/")
for child in res.json():
    st.image(child["image"], width=150)
    st.write(f"**Name:** {child['name']}")
    st.write(f"**Age:** {child['age']} | **Contact:** {child['contact']}")
    st.divider()

if st.button("Start Recognition (Local Camera)"):
    st.write("Run: `python core/realtime_recognition.py` in terminal.")
