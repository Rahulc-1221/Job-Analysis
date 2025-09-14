import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(layout="wide", page_title="Data Science Salaries Analytics", page_icon="💼")

# ----------------------------
# LOTTIE ANIMATION LOADER
# ----------------------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Jobs-related Lottie animation
lottie_url = "https://lottie.host/3167d8c8-188d-4e16-a4e7-0e53771f8a79/F1KpJcGuq9.json"
lottie_json = load_lottie_url(lottie_url)

# ----------------------------
# SIDEBAR MENU
# ----------------------------
with st.sidebar:
    if lottie_json:
        st_lottie(lottie_json, height=150)
    selected_menu = option_menu(
        menu_title="Menu",
        options=["Home", "Dataset Overview", "Salary Analysis", "Job Role Analysis", "Geographical Analysis", "Experience Analysis", "Company Insights"],
        icons=["house", "database", "currency-dollar", "briefcase", "globe", "graph-up", "building"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("D:\old datasets\jobs.csv")
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Map experience level codes to full names
    exp_map = {
        'EN': 'Entry-level',
        'MI': 'Mid-level',
        'SE': 'Senior-level',
        'EX': 'Executive'
    }
    df['experience_level'] = df['experience_level'].map(exp_map)
    
    # Map employment type codes to full names
    emp_map = {
        'FT': 'Full-time',
        'PT': 'Part-time',
        'CT': 'Contract',
        'FL': 'Freelance'
    }
    df['employment_type'] = df['employment_type'].map(emp_map)
    
    # Map company size codes to full names
    size_map = {
        'S': 'Small',
        'M': 'Medium',
        'L': 'Large'
    }
    df['company_size'] = df['company_size'].map(size_map)
    
    return df

df = load_data()

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("viridis")

# ----------------------------
# HOME PAGE
# ----------------------------
if selected_menu == "Home":
    st.title("💼 Data Science Salaries Analytics Dashboard")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Welcome to the Data Science Salaries Dashboard")
        st.write("""
        This interactive dashboard provides comprehensive insights into data science salaries across different roles,
        experience levels, locations, and company sizes. Explore the various sections to understand:
        
        - 📊 **Dataset Overview**: Summary statistics and data distribution
        - 💰 **Salary Analysis**: Salary trends and distributions
        - 👔 **Job Role Analysis**: Comparison of different data roles
        - 🌍 **Geographical Analysis**: Salary variations by location
        - 📈 **Experience Analysis**: How experience impacts compensation
        - 🏢 **Company Insights**: Company size and remote work trends
        """)
        
        st.info("💡 Use the sidebar to navigate between different sections of the dashboard.")
    
    with col2:
        if lottie_json:
            st_lottie(lottie_json, height=300, key="home_lottie")
    
    # Quick stats
    st.subheader("📈 Quick Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Job Titles", df['job_title'].nunique())
    with col3:
        st.metric("Average Salary (USD)", f"${df['salary_in_usd'].mean():,.0f}")
    with col4:
        st.metric("Data Years", f"{df['work_year'].min()} - {df['work_year'].max()}")

# ----------------------------
# DATASET OVERVIEW
# ----------------------------
elif selected_menu == "Dataset Overview":
    st.title("📊 Dataset Overview")
    
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Data Summary", "Missing Values"])
    
    with tab1:
        st.subheader("First Look at the Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
        
        with col2:
            st.subheader("Column Names")
            st.write(list(df.columns))
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Categorical Variables Summary")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            st.write(f"**{col}**: {df[col].nunique()} unique values")
            st.dataframe(df[col].value_counts().head(), use_container_width=True)
    
    with tab3:
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Values': missing_data.values,
            'Percentage': (missing_data.values / len(df)) * 100
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            fig = px.bar(missing_df, x='Column', y='Percentage', 
                         title='Percentage of Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in the dataset!")

# ----------------------------
# SALARY ANALYSIS
# ----------------------------
elif selected_menu == "Salary Analysis":
    st.title("💰 Salary Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Salary Distribution", "Salary Trends", "Salary by Company Size"])
    
    with tab1:
        st.subheader("Salary Distribution in USD")
        
        col1, col2 = st.columns(2)
        with col1:
            # Histogram
            fig = px.histogram(df, x='salary_in_usd', nbins=50, 
                              title='Distribution of Salaries (USD)',
                              labels={'salary_in_usd': 'Salary in USD'})
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y='salary_in_usd', title='Salary Distribution (USD)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Salary statistics
        st.subheader("Salary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Salary", f"${df['salary_in_usd'].mean():,.0f}")
        with col2:
            st.metric("Median Salary", f"${df['salary_in_usd'].median():,.0f}")
        with col3:
            st.metric("Minimum Salary", f"${df['salary_in_usd'].min():,.0f}")
        with col4:
            st.metric("Maximum Salary", f"${df['salary_in_usd'].max():,.0f}")
    
    with tab2:
        st.subheader("Salary Trends Over Time")
        
        # Average salary by year
        salary_by_year = df.groupby('work_year')['salary_in_usd'].agg(['mean', 'median', 'count']).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add mean salary line
        fig.add_trace(
            go.Scatter(x=salary_by_year['work_year'], y=salary_by_year['mean'], 
                      name="Mean Salary", mode='lines+markers', line=dict(width=3)),
            secondary_y=False,
        )
        
        # Add median salary line
        fig.add_trace(
            go.Scatter(x=salary_by_year['work_year'], y=salary_by_year['median'], 
                      name="Median Salary", mode='lines+markers', line=dict(width=3)),
            secondary_y=False,
        )
        
        # Add count as bars
        fig.add_trace(
            go.Bar(x=salary_by_year['work_year'], y=salary_by_year['count'], 
                   name="Number of Records", opacity=0.3),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text="Salary Trends Over Time",
            xaxis_title="Year",
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Salary (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Records", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data table
        st.dataframe(salary_by_year, use_container_width=True)
    
    with tab3:
        st.subheader("Salary by Company Size")
        
        # Salary by company size
        fig = px.box(df, x='company_size', y='salary_in_usd', 
                    title='Salary Distribution by Company Size',
                    color='company_size')
        st.plotly_chart(fig, use_container_width=True)
        
        # Average salary by company size
        avg_salary_size = df.groupby('company_size')['salary_in_usd'].mean().reset_index()
        fig = px.bar(avg_salary_size, x='company_size', y='salary_in_usd',
                    title='Average Salary by Company Size',
                    labels={'salary_in_usd': 'Average Salary (USD)', 'company_size': 'Company Size'})
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# JOB ROLE ANALYSIS
# ----------------------------
elif selected_menu == "Job Role Analysis":
    st.title("👔 Job Role Analysis")
    
    # Top job roles selector
    top_n = st.slider("Select number of top jobs to display:", 5, 30, 15)
    
    # Get top job roles by count
    job_counts = df['job_title'].value_counts().head(top_n)
    
    tab1, tab2, tab3 = st.tabs(["Job Frequency", "Salary by Job Role", "Job Role Details"])
    
    with tab1:
        st.subheader(f"Top {top_n} Most Common Job Titles")
        
        fig = px.bar(x=job_counts.values, y=job_counts.index, 
                     orientation='h', 
                     title=f'Top {top_n} Most Common Job Titles',
                     labels={'x': 'Count', 'y': 'Job Title'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Salary by Job Role")
        
        # Calculate average salary for top jobs
        top_jobs = job_counts.index.tolist()
        salary_by_job = df[df['job_title'].isin(top_jobs)].groupby('job_title')['salary_in_usd'].agg(['mean', 'median', 'count']).reset_index()
        salary_by_job = salary_by_job.sort_values('mean', ascending=False)
        
        fig = px.bar(salary_by_job, x='mean', y='job_title', 
                     orientation='h', 
                     title=f'Average Salary for Top {top_n} Job Titles',
                     labels={'mean': 'Average Salary (USD)', 'job_title': 'Job Title'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data
        st.dataframe(salary_by_job, use_container_width=True)
    
    with tab3:
        st.subheader("Explore Specific Job Role")
        
        selected_job = st.selectbox("Select a job title to explore:", df['job_title'].unique())
        
        job_data = df[df['job_title'] == selected_job]
        
        if not job_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Records", len(job_data))
                st.metric("Average Salary", f"${job_data['salary_in_usd'].mean():,.0f}")
                st.metric("Median Salary", f"${job_data['salary_in_usd'].median():,.0f}")
            
            with col2:
                # Experience level distribution for this job
                exp_dist = job_data['experience_level'].value_counts()
                fig = px.pie(values=exp_dist.values, names=exp_dist.index, 
                            title=f'Experience Level Distribution for {selected_job}')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show sample records
            st.dataframe(job_data, use_container_width=True)
        else:
            st.warning("No data available for the selected job title.")

# ----------------------------
# GEOGRAPHICAL ANALYSIS
# ----------------------------
elif selected_menu == "Geographical Analysis":
    st.title("🌍 Geographical Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Company Locations", "Employee Residences", "Remote Work Analysis"])
    
    with tab1:
        st.subheader("Company Locations Analysis")
        
        # Top company locations
        top_locations = df['company_location'].value_counts().head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(x=top_locations.values, y=top_locations.index, 
                         orientation='h', 
                         title='Top 15 Company Locations',
                         labels={'x': 'Count', 'y': 'Location'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average salary by company location (for top locations)
            salary_by_location = df[df['company_location'].isin(top_locations.index)].groupby('company_location')['salary_in_usd'].mean().reset_index()
            salary_by_location = salary_by_location.sort_values('salary_in_usd', ascending=False)
            
            fig = px.bar(salary_by_location, x='salary_in_usd', y='company_location', 
                         orientation='h', 
                         title='Average Salary by Company Location (Top 15)',
                         labels={'salary_in_usd': 'Average Salary (USD)', 'company_location': 'Location'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Employee Residences Analysis")
        
        # Top employee residences
        top_residences = df['employee_residence'].value_counts().head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(x=top_residences.values, y=top_residences.index, 
                         orientation='h', 
                         title='Top 15 Employee Residences',
                         labels={'x': 'Count', 'y': 'Residence'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average salary by residence (for top residences)
            salary_by_residence = df[df['employee_residence'].isin(top_residences.index)].groupby('employee_residence')['salary_in_usd'].mean().reset_index()
            salary_by_residence = salary_by_residence.sort_values('salary_in_usd', ascending=False)
            
            fig = px.bar(salary_by_residence, x='salary_in_usd', y='employee_residence', 
                         orientation='h', 
                         title='Average Salary by Employee Residence (Top 15)',
                         labels={'salary_in_usd': 'Average Salary (USD)', 'employee_residence': 'Residence'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Remote Work Analysis")
        
        # Remote ratio distribution
        remote_counts = df['remote_ratio'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=remote_counts.values, names=remote_counts.index, 
                        title='Distribution of Remote Work Ratio')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average salary by remote ratio
            salary_by_remote = df.groupby('remote_ratio')['salary_in_usd'].mean().reset_index()
            
            fig = px.bar(salary_by_remote, x='remote_ratio', y='salary_in_usd', 
                         title='Average Salary by Remote Work Ratio',
                         labels={'remote_ratio': 'Remote Ratio (%)', 'salary_in_usd': 'Average Salary (USD)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Map remote ratio to categories for better visualization
        df['remote_category'] = pd.cut(df['remote_ratio'], 
                                      bins=[-1, 0, 50, 99, 100], 
                                      labels=['No Remote', 'Partially Remote', 'Mostly Remote', 'Fully Remote'])
        
        salary_by_remote_cat = df.groupby('remote_category')['salary_in_usd'].mean().reset_index()
        
        fig = px.bar(salary_by_remote_cat, x='remote_category', y='salary_in_usd', 
                     title='Average Salary by Remote Work Category',
                     labels={'remote_category': 'Remote Work Category', 'salary_in_usd': 'Average Salary (USD)'})
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# EXPERIENCE ANALYSIS
# ----------------------------
elif selected_menu == "Experience Analysis":
    st.title("📈 Experience Level Analysis")
    
    tab1, tab2 = st.tabs(["Salary by Experience", "Experience Trends"])
    
    with tab1:
        st.subheader("Salary by Experience Level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot of salary by experience level
            fig = px.box(df, x='experience_level', y='salary_in_usd', 
                        color='experience_level',
                        title='Salary Distribution by Experience Level')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot for better distribution view
            fig = px.violin(df, x='experience_level', y='salary_in_usd', 
                           color='experience_level',
                           title='Salary Distribution by Experience Level (Violin Plot)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Average salary by experience level
        exp_salary = df.groupby('experience_level')['salary_in_usd'].agg(['mean', 'median', 'count']).reset_index()
        
        fig = px.bar(exp_salary, x='experience_level', y='mean', 
                    title='Average Salary by Experience Level',
                    labels={'mean': 'Average Salary (USD)', 'experience_level': 'Experience Level'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data
        st.dataframe(exp_salary, use_container_width=True)
    
    with tab2:
        st.subheader("Experience Level Trends Over Time")
        
        # Experience level distribution by year
        exp_by_year = pd.crosstab(df['work_year'], df['experience_level'])
        
        fig = px.line(exp_by_year, x=exp_by_year.index, y=exp_by_year.columns,
                     title='Experience Level Distribution Over Time',
                     labels={'value': 'Count', 'work_year': 'Year', 'variable': 'Experience Level'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Stacked area chart
        fig = px.area(exp_by_year, x=exp_by_year.index, y=exp_by_year.columns,
                     title='Experience Level Distribution Over Time (Stacked)',
                     labels={'value': 'Count', 'work_year': 'Year', 'variable': 'Experience Level'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data
        st.dataframe(exp_by_year, use_container_width=True)

# ----------------------------
# COMPANY INSIGHTS
# ----------------------------
elif selected_menu == "Company Insights":
    st.title("🏢 Company Insights")
    
    tab1, tab2 = st.tabs(["Company Size Analysis", "Employment Type Analysis"])
    
    with tab1:
        st.subheader("Company Size Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Company size distribution
            size_counts = df['company_size'].value_counts()
            fig = px.pie(values=size_counts.values, names=size_counts.index, 
                        title='Company Size Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average salary by company size
            salary_by_size = df.groupby('company_size')['salary_in_usd'].mean().reset_index()
            
            fig = px.bar(salary_by_size, x='company_size', y='salary_in_usd', 
                         title='Average Salary by Company Size',
                         labels={'company_size': 'Company Size', 'salary_in_usd': 'Average Salary (USD)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Company size distribution by year
        size_by_year = pd.crosstab(df['work_year'], df['company_size'])
        
        fig = px.line(size_by_year, x=size_by_year.index, y=size_by_year.columns,
                     title='Company Size Distribution Over Time',
                     labels={'value': 'Count', 'work_year': 'Year', 'variable': 'Company Size'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Employment Type Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Employment type distribution
            emp_counts = df['employment_type'].value_counts()
            fig = px.pie(values=emp_counts.values, names=emp_counts.index, 
                        title='Employment Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average salary by employment type
            salary_by_emp = df.groupby('employment_type')['salary_in_usd'].mean().reset_index()
            
            fig = px.bar(salary_by_emp, x='employment_type', y='salary_in_usd', 
                         title='Average Salary by Employment Type',
                         labels={'employment_type': 'Employment Type', 'salary_in_usd': 'Average Salary (USD)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Employment type distribution by year
        emp_by_year = pd.crosstab(df['work_year'], df['employment_type'])
        
        fig = px.line(emp_by_year, x=emp_by_year.index, y=emp_by_year.columns,
                     title='Employment Type Distribution Over Time',
                     labels={'value': 'Count', 'work_year': 'Year', 'variable': 'Employment Type'})
        st.plotly_chart(fig, use_container_width=True) 