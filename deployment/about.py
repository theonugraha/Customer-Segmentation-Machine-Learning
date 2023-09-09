import streamlit as st
from PIL import Image

def run():
    # Add Picture
    image = Image.open('cs.jpeg')
    st.image(image, caption=None, width=700)
    # Title
    st.title('ABOUT THIS PROJECT')
    st.markdown('---')
    st.header('Project Background')
    st.write('###### The financial industry has a large number of customers with diverse preferences, behaviors, and profiles. To improve service, customer retention and operational efficiency, financial companies need to understand their customers more deeply. One effective approach in achieving such understanding is to use clustering-based customer segmentation techniques. In this context, financial companies have access to customer data that includes information such as financial transactions, payment history, types of products used, and more. This data has great potential to reveal valuable insights into customer behavior, preferences, and potential risks.')
    st.markdown('---')
    st.header('Objective')
    st.write('###### The goal of this project is to implement a clustering-based customer segmentation technique using machine learning in the financial industry.')
    st.markdown('---')
    
    st.write('Feel free to contact me on:')
    st.write('[GITHUB](https://github.com/theonugraha)')
    st.write('or')
    st.write('[LINKEDIN](https://www.linkedin.com/in/nugrahatheo/)')


if __name__ == '__main__':
    run()