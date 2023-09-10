# Introduction
Name    : Theo Nugraha

Batch   : FTDS-RMT-021

# Dashboard/Deployment Link
Deployment Link  : [Customer Segmentation](https://huggingface.co/spaces/nugrahatheo/Customer-Segmentation)


# Project Title
**Cluster-based customer segmentation techniques using machine learning in the financial industry**

# About Dataset
[SOURCE DATASET](https://www.kaggle.com/datasets/sidharth178/customer-segmentation)

The sample Dataset summarizes the usage behavior of about nearly 1000 active credit card holders during the last 6 months. The file is at a customer level with 10 behavioral variables.

The following is the Data Dictionary for the Customer Segmentation dataset:

- `Customer Id`: This is typically a unique identifier assigned to each customer or individual in a dataset. It is used to distinguish and track individual customers. In credit analysis, it may not be a directly relevant factor for assessing creditworthiness, but it helps in keeping records and managing customer data.

- `Age`: This refers to the age of the customer or individual. Age can be an important factor in credit analysis because it is often correlated with financial stability and the ability to repay debts. Younger individuals may have less established credit histories, while older individuals may have more stable financial situations.

- `Edu` (Education): This likely represents the education level of the customer. Education can be a factor in assessing credit risk, as individuals with higher education levels may have higher earning potential and financial stability. It may also indicate a level of financial literacy.

- `Years Employed`: This represents the number of years the customer has been employed. Employment history is important in credit analysis because it reflects the stability of a person's income source. Longer periods of employment are generally viewed more favorably.

- `Income`: This is the customer's income, typically expressed as an annual figure. Income is a fundamental factor in credit analysis, as it determines a person's ability to make payments on debts. Higher income levels are generally associated with lower credit risk.

- `Card Debt`: Card debt refers to the amount of debt a customer has accumulated through the use of credit cards. It is a specific type of debt that can impact a person's creditworthiness.

- `Other Deb`: Other debt includes any additional debts or financial obligations that the customer may have, such as loans, mortgages, or personal debts. Total debt obligations are crucial in assessing a person's debt management capacity.

- `Defaulted`: "Defaulted" typically refers to whether the customer has failed to meet their debt obligations, such as missing payments or not repaying loans as agreed. It's a critical indicator of credit risk.

- `Debt-Income Ratio` (Debt-to-Income Ratio): This is a financial metric that measures the proportion of a customer's income that goes toward paying their debts. It is calculated by dividing the total debt (including card debt, other debt, etc.) by the customer's income. A high debt-to-income ratio can be a warning sign of financial stress and potential credit risk.

These variables are often used in credit scoring models and credit risk assessment to determine an individual's creditworthiness and the likelihood of them repaying debts as agreed. The specific importance of each variable may vary depending on the credit analysis model and the lender's criteria.

# Project Description
## Project Background
The financial industry has a large number of customers with diverse preferences, behaviors, and profiles. To improve service, customer retention and operational efficiency, financial companies need to understand their customers more deeply. One effective approach in achieving such understanding is to use clustering-based customer segmentation techniques.

In this context, financial companies have access to customer data that includes information such as financial transactions, payment history, types of products used, and more. This data has great potential to reveal valuable insights into customer behavior, preferences, and potential risks.

## Objective
The aim of this project is to implement cluster-based customer segmentation techniques (clustering) using machine learning in the financial industry. Some of the specific objectives to be achieved through this project include:

- **Customer Segmentation**: Identify different customer segments based on salary patterns, credit card debt, other debt and customer salary debt ratio.

- **Segment Profile**: Build a profile for each customer segment that includes key characteristics such as average customer salary, average customer credit card debt, average other debt, customer age and credit risk.

- **Behavior Analysis**: Analyze customer behavior within each segment, including salary trends, credit card debt trends, other debt trends and payroll debt ratio trends.

- **Personalization of Services**: Understand customer preferences within each segment to personalize services, offer appropriate products and increase retention.

- **Improved Operational Efficiency**: Optimize resource allocation by focusing on customer segments that have significant business impact.

- **Risk Management**: Identify customer segments that have high credit risks or suspicious behavior to take appropriate preventive measures.

By achieving these goals, financial companies can increase their understanding of customers, improve service, reduce risk, and achieve competitive advantage in a competitive industry.

## EDA (Exploratory Data Analysis)
![Edu](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/edu_plot.png)
![Defaulted](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/defaulted_plot.png)
![AIA](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_income_age.png)
![AICD](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_carddebt_age.png)
![AIOD](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_otherdebt_age.png)
![AIDIR](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_debtincomeratio_age.png)
![AYEI](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_yearsemployed_income.png)
![AYECD](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_yearsemployed_carddebt.png)
![AYEOD](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_yearsemployed_otherdebt.png)
![AYEDIR](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/av_yearsemployed_debtincomeratio.png)

## Clustering Method
The results of the clustering analysis using K-means indicate the existence of two distinct clusters with diverse financial characteristics. Each cluster represents a different customer profile, and the associated financial attributes provide valuable insights for customizing product and service offerings in the banking sector.

![Viz](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/cluster_viz.png)
## Cluster Characteristics Results
![Heatmap Clustering](https://github.com/theonugraha/Customer-Segmentation-Machine-Learning/blob/main/Readme%20Contents/heatmap%20clustering.png)

**Cluster 0**:

- Age: Low
- Years Employed: Low
- Income: Low
- Card Debt: Low
- Other Debt: Low
- Debt Income Ratio: Low
- Insights for Cluster 0: This cluster includes customers with a low financial profile, including young age, low income and low debt. Customers in this cluster may need banking products that focus on initial financial development, such as savings accounts, checking accounts, or savings products with competitive interest rates. They may also need access to basic banking services such as debit cards to facilitate everyday transactions.

**Cluster 1**:

- Age: High
- Years Employed: High
- Income (Income): High
- Card Debt (Credit Card Debt): High
- Other Debt: High
- Debt Income Ratio: High
- Insights for Cluster 1: This cluster consists of customers with a strong financial profile, including higher age, high income, and high debt, especially in the form of credit cards and other debt. Customers in this cluster may be more interested in banking products which tend to be more complex and oriented towards investment or wealth management. Products that suit them may include stock investments, mutual funds, retirement accounts, or portfolio management services.


# Conclusion
From the business side, we can draw conclusions that cover several important aspects related to customer groups in Cluster 0 and Cluster 1, namely age, income, credit card debt, other debt, and debt to income ratio. The following are conclusions that represent these aspects:

**Customer Segmentation**

In this customer segmentation, there are two main clusters (Cluster 0 and Cluster 1) which differentiate customers based on their age (Age Group) and years of work experience (Years Employed).

**Segment Profile**

- **Cluster 0**: This cluster includes most customers with ages ranging from 20s to 50s. Customers in this cluster have low to high average income (Average Income), low to medium average card debt (Average Card Debt), low to medium average other debt (Average Other Debt), and average -The average debt to income ratio (Average Debt Income Ratio) is relatively low to moderate. They also have varying years of work experience, from 0 to 30 years.

- **Cluster 1**: This cluster consists of customers with a similar age to Cluster 0, which ranges from 20s to 50s. However, customers in this cluster have a higher average income (Average Income), a higher average card debt (Average Card Debt), a higher average other debt (Average Other Debt), and an average of a higher average debt to income ratio. They also have varying years of work experience, ranging from 0 to 31+ years.

**Behavior Analysis**

- Customers in Cluster 0 tend to have lower levels of debt, both card debt and other debt, and a lower debt ratio compared to Cluster 1.
- Customers in Cluster 1 tend to have higher incomes, but also higher levels of debt, especially in terms of card debt and other debt.

**Personalization of Services**

- Based on this segmentation, banks or financial institutions can customize their services and products for each cluster. Cluster 0 may be more interested in products that reduce debt ratios or savings-based products, while Cluster 1 may look for investment or credit products with higher limits.

**Risk management**

For risk management, banks can pay more attention to parameters from these two clusters. Cluster 1, with higher levels of debt, may have higher credit risks, so banks need to be more careful in assessing loan suitability for them. Meanwhile, Cluster 0 may have lower credit risk, but other factors such as years of work experience also need to be taken into account in assessing their risk. With a better understanding of customer profiles, banks can optimize their risk management.

**Recommendations for Banking Products or Financial Products**:

Based on the conclusions above, here are several recommendations for banking products or financial products that are suitable for these two customer groups:

**Cluster 0**:

- **Savings** and **Investments**: Encourage them to take advantage of higher incomes as they age by offering savings or investment products that help them increase their wealth.

- **Mutual Fund Products**: Suggest mutual fund products that suit their risk profile to help them invest in a diversified manner.

- **Low Interest Rate Credit**: Offers credit products to consumers with low interest rates or credit cards with competitive interest rates to help them manage their daily financial needs.

**Cluster 1**:

- **Debt Management**: Offers debt management and financial consulting services that can help them manage their debt better.

- **Consolidation Credit**: Recommend debt consolidation products that help them consolidate debt with lower interest rates, thereby reducing debt burden.

- **Income Protection Insurance**: Provides income protection insurance products or credit insurance that can protect them in emergency situations and help them pay off their debts if the unexpected happens.

- **Financial Education Program**: Provides a special financial education program for Cluster 1 members so they can understand how to manage debt better and avoid accumulating greater debt.

- **Debt Recovery Investment**: Offers investment products that can help them earn additional income to pay off their debt faster.

It is always important to understand individual customer profiles and their needs in more detail before offering a particular banking or financial product. Additionally, an approach that focuses on financial education can also help.
