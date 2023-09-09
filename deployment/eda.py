import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title = 'CUSTOMER SEGMENTATION MACHINE LEARNING',
    layout ='centered',
    initial_sidebar_state='expanded'
)

def run():
    # Sub Header
    st.subheader('EDA for Analizing Customer Segmentation', )

    # Separated Line
    st.markdown('---')

    # Show Data Frame
    st.write('#### Dataset Customer Segmentation')
    data = pd.read_csv('customer_segmentation.csv')
    st.dataframe(data)

    # Histogram based user input
    st.write('#### Histogram')
    option = st.selectbox('Choose Column : ', ('age','creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
                                                'serum_creatinine', 'serum_sodium', 'time'))
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(data[option], bins=30, kde=True)
    st.pyplot(fig)
    
    # Pie chart `DEATH_EVENT`
    death_data = data["DEATH_EVENT"].value_counts()
    st.write('#### Pie Chart DEATH EVENT')
    fig = plt.figure(figsize=(6, 6))
    plt.pie(death_data, labels=["Not Death", "Death"], autopct="%1.1f%%", startangle=90, colors=["lightblue", "lightcoral"])
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can understand that the surviving patients outnumber the deceased patients. It also shows that the proportion of the two classes is not balanced.
            ''')
    
    # Pie chart `sex`
    sex_data = data["sex"].value_counts()
    st.write('#### Pie Chart SEX')
    fig = plt.figure(figsize=(6, 6))
    plt.pie(sex_data, labels=["Woman", "Man"], autopct="%1.1f%%", startangle=90, colors=["lightgreen","lightblue"])
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can understand that the woman patients outnumber the man patients. It also shows that the proportion of the two classes is not balanced.
            ''')
        
    # Pie chart `diabetes`
    diabetes_data = data["diabetes"].value_counts()
    st.write('#### Pie Chart DIABETES')
    fig = plt.figure(figsize=(6, 6))
    plt.pie(diabetes_data, labels=["No Diabetes", "Diabetes"], autopct="%1.1f%%", startangle=90, colors=["lightgreen","red"])
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can know that patients who do not have diabetes are more than patients who have diabetes.
            ''')
        
    # Pie chart `anaemia`
    anaemia_data = data["anaemia"].value_counts()
    st.write('#### Pie Chart ANAEMIA')
    fig = plt.figure(figsize=(6, 6))
    plt.pie(anaemia_data, labels=["No Anaemia", "Anaemia"], autopct="%1.1f%%", startangle=90, colors=["lightyellow","red"])
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can know that patients who do not have anaemia are more than patients who have anaemia.
            ''')
        
    # Pie chart `high_blood_pressure`
    hbp_data = data["high_blood_pressure"].value_counts()
    st.write('#### Pie Chart HIGH BLOOD PRESSURE')
    fig = plt.figure(figsize=(6, 6))
    plt.pie(hbp_data, labels=["No High Blood Pressure", "High Blood Pressure"], autopct="%1.1f%%", startangle=90, colors=["lightblue","red"])
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can know that patients who do not have high blood pressure are more than patients who have high blood pressure.
            ''')
    
    # Pie chart `smoking`
    smoking_data = data["smoking"].value_counts()
    st.write('#### Pie Chart SMOKING')
    fig = plt.figure(figsize=(6, 6))
    plt.pie(smoking_data, labels=["No Smoking", "Smoking"], autopct="%1.1f%%", startangle=90, colors=["lightgrey","red"])
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can know that patients who do not smoke are more than patients who smoke.
            ''')

    # Bar Plot 1
    st.write('#### Plot Death Event Based on Sex')
    fig   = plt.figure(figsize=(15, 5))
    death = sns.countplot(data=data, x="sex", hue="DEATH_EVENT")
    for container in death.containers:
        death.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can see that more man patients die than woman patients.
            ''')
        
    # Bar Plot 2
    st.write('#### Plot Death Event Based on Smoking')
    fig   = plt.figure(figsize=(15, 5))
    smoking = sns.countplot(data=data, x="smoking", hue="DEATH_EVENT")
    for container in smoking.containers:
        smoking.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can see that the number of patients who do not smoke and die is equal to the number of patients who smoke and do not die.
            ''')
    
    # Bar Plot 3
    st.write('#### Plot Death Event Based on Anaemia')
    fig   = plt.figure(figsize=(15, 5))
    anaemia = sns.countplot(data=data, x="anaemia", hue="DEATH_EVENT")
    for container in anaemia.containers:
        anaemia.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can see that patients who do not die with non-anaemia are more than patients who die with anaemia.
            ''')
    
    # Bar Plot 4
    st.write('#### Plot Death Event Based on Diabetes')
    fig   = plt.figure(figsize=(15, 5))
    diabetes = sns.countplot(data=data, x="diabetes", hue="DEATH_EVENT")
    for container in diabetes.containers:
        diabetes.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can see that patients who do not die with non-diabetics are more than patients who die with diabetes.
            ''')
        
    # Bar Plot 5
    st.write('#### Plot Death Event Based on High Blood Pressure')
    fig   = plt.figure(figsize=(15, 5))
    hbp = sns.countplot(data=data, x="high_blood_pressure", hue="DEATH_EVENT")
    for container in hbp.containers:
        hbp.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can see that patients who do not die with non-high blood pressure are more than patients who die with high blood pressure.
            ''')

if __name__ == '__main__':
    run()