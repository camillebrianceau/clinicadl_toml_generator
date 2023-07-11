import streamlit as st

from form import Form



def main():

	st.sidebar.file_uploader("Some help to find paths.", accept_multiple_files=True)
	

	form = Form()
	form.create_page()
	

if __name__ == '__main__':
	main()
