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
    option = st.selectbox('Choose Column : ', ('Customer Id','Age', 'Years Employed', 'Income', 
                                                'Card Debt', 'Other Debt', 'DebtIncomeRatio'))
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(data[option], bins=30, kde=True)
    st.pyplot(fig)
    
    # Education Plot
    # Data for bar plot
    education = data["Edu"].value_counts()

    # Data for pie chart
    labels = ["1", "2", "3", "4", "5"]
    sizes = education.values
    colors = ["lightblue", "lightcoral", "lightgreen", "yellow", "lightgrey"]

    # Create a Streamlit app
    st.title("Education Analysis")

    # Plot bar chart
    st.subheader("Count of Education (Bar Plot)")
    plt.figure(figsize=(12, 6))
    plt.bar(labels, sizes, color=colors)
    # Adding a label to a bar
    for i, value in enumerate(education.values):
        plt.text(i, value, str(value), ha='center', va='bottom', fontsize=10)
    plt.xlabel("Education")
    plt.ylabel("Count")
    plt.title("Count of Education")
    st.pyplot(plt)  # Display the plot in Streamlit

    # Plot pie chart
    st.subheader("Proportion of Education (Pie Chart)")
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=150, colors=colors)
    plt.title("Proportion of Education")
    st.pyplot(plt)  # Display the plot in Streamlit
    with st.expander("See explanation"):
        st.write('''
            Generally, there are five main levels of education that are often recognized internationally. These levels of education are in order from lowest to highest and may vary from country to country, but are generally as follows:

            1. **Pre-school education**: This is the initial level of education aimed at young children, usually from 3 to 6 years of age. Pre-school education may take the form of kindergarten or other early education programs designed to help children develop their social, cognitive and motor skills before starting primary education.

            2. **Primary Education**: This level covers basic education provided to children in the age group of 6 to 12 years. This is the level of education that includes basic subjects such as language, math, science and social studies. In many countries, primary education is compulsory.

            3. **Secondary Education**: Secondary education (or secondary school) is the level of education provided to students in the age group of 12 to 18 years. It is often divided into two stages: first secondary education (junior high school) and second secondary education (senior high school). Students usually take more in-depth subjects at this level and prepare themselves for admission to college or employment.

            4. **Higher Education**: Higher education includes various educational programs at colleges and universities. It includes bachelor's degrees (S1), master's degrees (S2), and doctoral degrees (S3). Higher education allows students to gain more in-depth knowledge in a particular field of study and develop the skills necessary for various professions.

            5. **Professional and Technical Education**: This is an additional level of education that provides specialized training in certain fields that require practical expertise. It can include technical schools, vocational schools and other professional training programs. Students pursuing professional and technical education usually prepare for a specific career, such as technician, mechanic or nurse.

            If we refer to the education levels, here are some insights that can be gained from the plot:

            - **Distribution of Education Levels**: This data provides an overview of the distribution of education levels within the observed population or sample. In this case, education level 1 has the highest number (460 individuals), which indicates that most people in the dataset have education level 1. Education level 2 has a lower number (235 individuals), and so on.

            - **Majority with Low Education Level**: The majority of individuals in the dataset have lower education levels (1 and 2), while the number of individuals with higher education levels (3, 4, and 5) is decreasing significantly.

            - **Education Imbalance**: There is a significant imbalance in education levels in the dataset. This could have implications in further analysis, especially if education level is an important factor in the analysis or modeling.

            - **Low Numbers at Higher Education Levels**: Education levels 4 and 5 have very low numbers (49 and 5 individuals, respectively). This suggests that individuals with higher education levels in this dataset may be a minority group.
            ''')
    
    # Data for bar plot
    defaulted = data["Defaulted"].value_counts()

    # Data for pie chart
    labels = ["No Default", "Default"]
    sizes = defaulted.values
    colors = ["green", "red"]

    # Create a Streamlit app
    st.title("Defaulted Analysis")

    # Plot bar chart
    st.subheader("Bar Plot")
    fig_bar = plt.figure(figsize=(12, 6))
    plt.bar(labels, sizes, color=colors)
    # Adding a label to a bar
    for i, value in enumerate(defaulted.values):
        plt.text(i, value, str(value), ha='center', va='bottom', fontsize=10)
    plt.xlabel("Defaulted")
    plt.ylabel("Count")
    plt.title("Count of Defaulted")

    st.pyplot(fig_bar)

    # Plot pie chart
    st.subheader("Pie Chart")
    fig_pie = plt.figure(figsize=(12, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title("Proportion of Defaulted")

    st.pyplot(fig_pie)
    with st.expander("See explanation"):
        st.write('''
            Based on the results of the above plot, we can know that customers who do not default are more than customers who default. Business-wise, this is very good as the majority of customers do not default or pay their debts smoothly.
            ''')

    # Group the data by education level and calculate the average income in each group
    income_by_education = data.groupby('Edu')['Income'].mean().reset_index()

    # Sort data based education
    income_by_education = income_by_education.sort_values(by='Edu')

    # Set up Streamlit app
    st.title('Average Income by Education Analysis')

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(income_by_education['Edu'], income_by_education['Income'], color='blue')
    ax.set_xlabel('Education')
    ax.set_ylabel('Average Income')
    ax.set_title('Average Income by Education')

    # Add labels above each bar
    for bar, income in zip(bars, income_by_education['Income']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, round(income, 2), ha='center')

    # Display plot using Streamlit
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            - **Correlation between Education and Income**:

            It can be seen that average income tends to increase with higher levels of education. This is a generally expected pattern, where people with higher levels of education tend to have higher incomes. This could be due to better job opportunities and access to higher paying jobs for those with higher education levels.

            - **Higher Income for Education Level 5**:

            Education level 5 (presumably an advanced level of education, such as a bachelor's degree or higher) shows a significant average income of around 116,600. This indicates that those with higher education have significantly higher incomes compared to those with lower levels of education. This may also reflect the high demand for higher skills and knowledge in the job market.

            - **Significant Differences between Education Levels**:

            The difference in average earnings across education levels is significant. This highlights the importance of education in influencing individual earnings.
            Education level 1 (presumably primary education or equivalent) has the lowest average income of around 40,837, while education level 5 has the highest average income of around 116,600.

            - **The Importance of Education in Economic Wellbeing**:

            These results underscore the importance of education in achieving higher economic well-being. Investing in education can open doors to better job opportunities and higher earnings in the future.

            - **Impact on Decision-Making**:

            These results can help individuals and government policies to make better decisions about education and careers. People can see that higher education can contribute significantly to future earnings, while education and training policies can be designed to increase access to higher levels of education.
            ''')
        
    # Group the data by age and calculate the average income
    average_income_by_age = data.groupby('Age')['Income'].mean()

    # Create a Streamlit app
    st.title('Average Income by Age Analysis')

    # Create a plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(average_income_by_age.index, average_income_by_age.values, marker='o', linestyle='-')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Income')
    ax.set_title('Line Plot of Average Income by Age')
    ax.grid(True)
    # Display the plot using Streamlit
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Insights that can be obtained from data on average income based on age are as follows:

            - **Increase in Income with Increasing Age**:

            It can be seen that in this data, average income tends to increase with age. This is consistent with the general expectation that people tend to experience increases in income over time.
            This increase may be due to increased work experience, career advancement, and access to better jobs with age.

            - **Income Variability**:

            Although there is a general trend of increasing income with age, there is significant variation in average income across certain age groups.
            It can be seen that some age groups have much higher average incomes than others. For example, at age 51, there is a drastic jump in average income, reaching 120,857. Likewise, at age 43, there is a very high average income (90,273).

            - **High Income Outliers**:

            There are several age groups that may have outliers in income data. For example, at age 47, the average income reaches 99,818, which may be an outlier.
            Outliers in income data can be caused by factors such as a very successful career, investments, or unusual financial situations.

            - **Decline in Income After Career Peak**:

            After reaching the peak, it appears that there is a decline in average income in some age groups. For example, at age 55, there is a significant decline in average earnings after reaching 59. This could indicate retirement or a decline in earnings after a career peak.

            - **Income Differences at Different Stages of Life**:

            These data also reflect differences in income across life stages. At a young age, income tends to be low, then increases with age, peaks at a certain age, and then may decline.
                        ''')
    
    # Group data by age and calculate average card debt
    average_card_debt_by_age = data.groupby('Age')['Card Debt'].mean()

    # Create a Streamlit app
    st.title('Average Card Debt by Age Analysis')

    # Plot the data using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(average_card_debt_by_age.index, average_card_debt_by_age.values, marker='o', linestyle='-')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Card Debt')
    ax.set_title('Line Plot of Average Card Debt by Age')
    ax.grid(True)

    # Display the plot in the Streamlit app
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            From the data on average card debt based on age, there are several insights that can be obtained:

            - **Credit Card Debt Increases with Age**:

            It appears that credit card debt tends to increase with age initially, peaking at age 54 (with an average of 7.262) and then decreasing significantly at age 55 (to 2.484).
            This increase may reflect individual financial developments, such as greater use of credit cards for important transactions and purchases at certain ages.

            - **Fluctuations in Credit Card Debt**:

            At some point, there are fluctuations in credit card debt that may be caused by economic factors or changes in an individual's lifestyle.
            For example, between ages 44 and 45 there is a sharp decline in credit card debt, possibly due to more frugal spending or significant debt payments.

            - **Significant Peak Credit Card Debt at Age 54**:

            Age 54 represents the peak of credit card debt with an average of 7.262. This could be the point at which a person reaches a financial plateau or has certain financial obligations that impact credit card debt levels.

            - **Outliers at Age 47**:

            There is a significant outlier at age 47, where the average credit card debt is 4.244. This could be due to a unique financial situation or a change in financial needs at that age.

            - **Correlation with External Factors**:

            A significant increase in credit card debt at a certain age may be related to certain life events, such as purchasing a home, children's education, or retirement plans.
            An understanding of credit card debt trends by age can assist credit card companies, financial institutions, or individuals in their financial planning and credit card debt management.
                        ''')

    # Group data by age and calculate average other debt
    average_other_debt_by_age = data.groupby('Age')['Other Debt'].mean()

    # Tampilkan plot menggunakan Streamlit
    st.title('Average Other Debt by Age Analysis')

    # Buat plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(average_other_debt_by_age.index, average_other_debt_by_age.values, marker='o', linestyle='-')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Other Debt')
    ax.grid(True)

    # Tampilkan plot di Streamlit
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Insights from the average other debt data based on age are as follows:

            - **Increasing Debt with Early Age**:

            At early age (20-25 years), average other debts tend to increase significantly. This could reflect a period when an individual begins to take on financial responsibilities such as student loans or consumer debt.

            - **Stability in Middle Age (25-35 years)**:

            Between ages 25 and 35, average other debts remain relatively stable with smaller fluctuations. This may indicate that at this age, people have stabilized their financial situation and other debts tend to remain constant.
            Significant Improvement After Age 35:

            After age 35, there is a significant increase in average other debts. This increase can be caused by a variety of factors, including financial responsibilities such as mortgages, greater consumer debt, or investments in advanced education.

            - **Peak at Age 54**:

            Age 54 represents another peak in average debt, with the highest value being around 8,394. This may reflect a period of preparation for retirement, financing children's education, or the accumulation of debt from various sources.

            - **Variability at Specific Ages**:

            There are some fluctuations at certain ages, such as at ages 37 and 49, that may reflect significant financial events or changes at these points in a person's life.

            - **Certain Declines at Ages 55 and 56**:

            There appears to be a significant decrease in average other debt at ages 55 and 56, possibly reflecting preparation for retirement and debt reduction.
                        ''')
        
    # Group data by age and calculate average Debt Income Ratio
    average_DebtIncomeRatio_by_age = data.groupby('Age')['DebtIncomeRatio'].mean()

    # Create a Streamlit app
    st.title('Average Debt Income Ratio by Age Analysis')

    # Create a plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(average_DebtIncomeRatio_by_age.index, average_DebtIncomeRatio_by_age.values, marker='o', linestyle='-')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Debt Income Ratio')
    ax.set_title('Line Plot of Average Debt Income Ratio by Age')
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Insights into the average Debt Income Ratio based on age are as follows:

            - **Increase in Debt Ratio in Early Age (20-28 years)**:

            It can be seen that at an early age, especially between 20 and 28 years, the debt-to-income ratio tends to increase significantly. This could reflect the fact that individuals at this age may be just starting out in a financially independent life, including taking out credit or loans for education or purchasing assets such as a car or house.

            - **Declining Trend in Middle Age (29-38 years)**:

            Between the ages of 29 and 38, the debt-to-income ratio shows a downward trend. It may reflect a period when individuals were more financially stable, paid their debts and managed their finances better.
            Increased Return in Old Age (39-56 years):

            After age 38, there is another increase in the debt-to-income ratio, especially after age 39. This could be due to financial responsibilities such as a home mortgage or preparations for retirement.

            - **Peak at Age 54**:

            Age 54 represents a peak debt-to-income ratio with an average of around 16,050. This could reflect preparation towards retirement or the accumulation of debt for various financial goals.

            - **Variability at a Specific Point**:

            There are fluctuations in the ratio of debt to income at certain points, such as the age of 47 years who recorded the highest ratio of 12.5. It may indicate significant financial events or changes at this age.

            - **The Importance of Debt Management at Early and Late Ages**:

            The increase in debt ratios in early and late life highlights the importance of debt management. At an early age, individuals need to understand the risks and responsibilities associated with debt. In later life, wise financial planning and effective debt management can help reduce financial risks in retirement.

            - **Trends Correlated with Life Stage**:

            These data reflect trends based on life stage. At an early age, individuals may be more inclined to take financial risks for education or purchasing assets. In middle age, financial stability tends to be achieved. At an advanced age, there is preparation for retirement.

            - **The Importance of Debt Management Based on Age**:

            These results can be used to inform the banking and finance business approach to debt management and product offerings. Companies can provide services that are more tailored to the needs of individuals at certain life stages, such as retirement planning or debt management at a young age.
                        ''')

if __name__ == '__main__':
    run()