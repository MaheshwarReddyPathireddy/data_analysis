import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import plotly.express as px
from sklearn.linear_model import LinearRegression


st.set_page_config(layout="wide")
st.title("Analyzing Birth Rate And Death Rate Trends Over The Past Decade Based On Data")

col1, col2 = st.columns([1, 2])
with col1:
    image_url = "/Users/maheshwarreddy/Downloads/indiamap.jpeg"
    st.image(image_url, caption="")

with col2:
    data = {
        "Metric": [
            "Population",
            "Current health expenditure (% of GDP)",
            "WHO region",
            "World Bank income level"
        ],
        "Value": [
            "1,417,173,167 (2022)",
            "3.28 (2021)",
            "South East Asia",
            "Lower-middle income (LMC)"
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)

st.markdown("""
The population figure is as of 2022.
The expenditure figure is as of 2021.
""")
st.subheader("Select a  DEMOGRAPHIC category to view")

categories = {
    "AGE DISTRIBUTION OF POPULATION": [],
    "POPULATION TREND": [],
    "ALCOHOL": [],
    "HEALTHIER POPULATION":[],
    "TOBACCO CONSUMERS AGED 15 AND OLDER": [],
    "HEALTH EMERGENCIES PROJECTION ":[],
    "DEMOGRAPHIC CHANGE":[],
    "PREDICTED BIRTH RATE":[],
    "PREDICTED DEATH RATE":[],
    "CAUSE OF DEATHS":[],
    "BIRTH RATE ANALYSIS":[],
    "DEATH RATE ANALYSIS":[],

}

selected_category = st.selectbox("Demographic Category", list(categories.keys()))

if selected_category:
    if selected_category=="PREDICTED DEATH RATE":
        st.title("PREDICTED DEATH RATE OF INDIA FOR THE NEXT 15 YEARS")
       
        years = [1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        deaths = [17, 15, 14, 13, 14, 14, 13, 13, 13, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 8, 9, 8]
        X = np.array(years).reshape(-1, 1)
        y = np.array(deaths)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        future_years = np.array(range(2021, 2041)).reshape(-1, 1)
        future_years_poly = poly_features.transform(future_years)
        predicted_deaths = model.predict(future_years_poly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=deaths,
            mode='markers',
            name='Historical Data',
            marker=dict(size=8, color='blue'),
            hovertemplate='Year: %{x}<br>Deaths: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=list(range(2021, 2041)),
            y=predicted_deaths,
            mode='lines+markers',
            name='Predicted Data',
            line=dict(color='red'),
            marker=dict(size=8, color='red'),
            hovertemplate='Year: %{x}<br>Predicted Deaths: %{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            title='Deaths Over the Years (Polynomial Regression)',
            xaxis_title='Year',
            yaxis_title='Deaths',
            hovermode='x'
        )
        st.plotly_chart(fig)
    if selected_category=="DEATH RATE ANALYSIS":
        st.title("DEATH RATE PER 1000")
        years = [1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        deaths = [17, 15, 14, 13, 14,14, 13, 13, 13, 12, 12, 12,12, 12, 12, 11, 11, 11, 11, 11, 11,11, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 8, 9, 8]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=deaths,
            mode='lines+markers',
            name='Deaths',
            line=dict(color='blue'),
            marker=dict(size=8, color='red'),
            hovertemplate='Year: %{x}<br>Deaths: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title='Births Over the Years',
            xaxis_title='Year',
            yaxis_title='Deaths',
            hovermode='x'
        )

        st.plotly_chart(fig)
        
    elif selected_category == "BIRTH RATE ANALYSIS":
        st.title("BIRTH RATE PER 1000")
        years = [1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        births = [43, 42, 42, 42, 42, 41, 41, 41, 40, 40, 39, 39, 39, 39, 38, 38, 37, 37, 36, 36, 36, 36, 36, 36, 35, 35, 35, 34, 34, 33, 33, 32, 31, 30, 30, 29, 29, 28, 28, 28, 28, 28, 27, 27, 26, 25, 24, 23, 23, 22, 22, 21, 21, 20, 20, 19, 18, 17, 17, 16, 16]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=births,
            mode='lines+markers',
            name='Births',
            line=dict(color='blue'),
            marker=dict(size=8, color='red'),
            hovertemplate='Year: %{x}<br>Births: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title='Births Over the Years',
            xaxis_title='Year',
            yaxis_title='Births',
            hovermode='x'
        )

        st.plotly_chart(fig)

        
    if selected_category == "DEMOGRAPHIC CHANGE":
        age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
        female_2023 = [55.2,57.3,60,61,61.6,58.9,56.5,52.8,46.8,41.2,36,30.8,25.8,20.6,14.5,8.6,5,3.5]
        male_2023 = [59.6, 62.5, 65.9, 67.2, 68, 64.8, 61.3, 57.1, 50.3, 43.4, 37.3, 31.5, 25.9, 20, 13.5, 7.4, 4, 2.5]
        female_2050 = [48.9, 51, 53.3, 55.2, 56, 56, 59.4, 62.2, 63.6, 64.8,61.6, 56.4, 50.2, 41.5, 31.2, 21.9, 13.3, 9.8] 
        male_2050 = [ 46.6, 48.8, 50.6, 52.3, 52.7, 52.7, 54.8, 57, 58.4, 59.9, 57.3, 53.8, 49.2, 41.8, 33 ,24.6, 16.1,13.4]
        st.title("Population by Age and Sex - India")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2023")
            fig_2023 = go.Figure()

            fig_2023.add_trace(
                go.Bar(
                    y=age_groups,
                    x=female_2023,
                    name='Female',
                    orientation='h',
                    marker=dict(color='#FF69B4')
                )
            )
            fig_2023.add_trace(
                go.Bar(
                    y=age_groups,
                    x=male_2023,
                    name='Male',
                    orientation='h',
                    marker=dict(color='#00BFFF')
                )
            )

            fig_2023.update_layout(
                barmode='group',
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis=dict(
                    title_text='Age Groups',
                    gridcolor='lightgray',
                    gridwidth=1,
                    tickfont=dict(size=14, color='black')
                ),
                xaxis=dict(
                    title_text='Population',
                    gridcolor='lightgray',
                    gridwidth=1,
                    tickfont=dict(size=14, color='black')
                ),
                font=dict(
                    color='black',
                    size=16
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )

            st.plotly_chart(fig_2023, use_container_width=True)
        with col2:
            st.subheader("2050")
            fig_2050 = go.Figure()

            fig_2050.add_trace(
                go.Bar(
                    y=age_groups,
                    x=female_2050,
                    name='Female',
                    orientation='h',
                    marker=dict(color='#FF1493')
                )
            )
            fig_2050.add_trace(
                go.Bar(
                    y=age_groups,
                    x=male_2050,
                    name='Male',
                    orientation='h',
                    marker=dict(color='#1E90FF')
                )
            )

            fig_2050.update_layout(
                barmode='group',
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis=dict(
                    title_text='Age Groups',
                    gridcolor='lightgray',
                    gridwidth=1,
                    tickfont=dict(size=14, color='black')
                ),
                xaxis=dict(
                    title_text='Population',
                    gridcolor='lightgray',
                    gridwidth=1,
                    tickfont=dict(size=14, color='black')
                ),
                font=dict(
                    color='black',
                    size=16
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )

            st.plotly_chart(fig_2050, use_container_width=True)

        

    elif selected_category == "AGE DISTRIBUTION OF POPULATION":
      
# Data
        age_groups = ['0-14', '15-64', '65+']
        percentages = [12, 80, 8]

        # Streamlit App
        st.title("Age Distribution of Population (%) - India, 2023")

        # Plotly Bar Chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=age_groups,
                    y=percentages,
                    text=percentages,
                    textposition='auto',
                    marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                )
            ]
        )

        # Update layout for white background
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title_text='Age Groups',
                gridcolor='lightgray',
                gridwidth=1,
                tickfont=dict(size=14, color='black')
            ),
            yaxis=dict(
                title_text='Percentage (%)',
                gridcolor='lightgray',
                gridwidth=1,
                tickfont=dict(size=14, color='black'),
                range=[0, 100]
            ),
            font=dict(
                color='black',
                size=16
            )
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    elif selected_category == "POPULATION TREND":


# Data


# Data
        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
        population = [1_000_000_000, 1_020_000_000, 1_040_000_000, 1_060_000_000, 1_080_000_000, 1_100_000_000, 1_120_000_000, 1_140_000_000, 1_160_000_000, 1_180_000_000, 1_200_000_000, 1_220_000_000, 1_240_000_000, 1_240_000_000, 1_240_000_000, 1_240_000_000, 1_240_000_000,1_260_000_000, 1_280_000_000, 1_280_000_000, 1_300_000_000, 1_320_000_000, 1_340_000_000, 1_360_000_000, 1_380_000_000, 1_400_000_000, 1_420_000_000, 1_440_000_000, 1_460_000_000, 1_480_000_000, 1_500_000_000, 1_520_000_000, 1_540_000_000, 1_560_000_000, 1_580_000_000, 1_600_000_000, 1_620_000_000, 1_640_000_000, 1_660_000_000, 1_680_000_000, 1_700_000_000, 1_700_000_000, 1_700_000_000, 1_760_000_000, 1_760_000_000, 1_760_000_000,1_760_000_000, 1_780_000_000, 1_780_000_000, 1_780_000_000, 1_780_000_000, ]
        # Ensure both lists have the same length
        
        # Create DataFrame
        df = pd.DataFrame({
            'Year': years,
            'Population': population
        })

        # Streamlit App
        st.title("Population Trend in India (2000-2050)")

        # Plotly Line Chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df['Year'],
                y=df['Population'],
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=10),
                hovertemplate='Year: %{x}<br>Population: %{y:,.0f}'
            )
        )

        # Update layout for white background and hover information
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title_text='Year',
                gridcolor='black',
                gridwidth=1,
                tickvals=years,
                ticktext=[str(year) for year in years],
                tickfont=dict(size=14),
                tickcolor='black'
            ),
            yaxis=dict(
                title_text='Population (in billions)',
                gridcolor='black',
                gridwidth=1,
                tickformat=',',
                range=[0, 2_000_000_000],
                tickvals=[1_000_000_000, 1_200_000_000, 1_400_000_000, 1_600_000_000, 1_800_000_000]
            ),
            font=dict(
                color='black',
                size=16
            ),
            hovermode='x unified'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
            
    elif selected_category == "CAUSE OF DEATHS":
        st.title("deaths per 100000")
        data = {
            'Cause': [
                'COVID-19', 'Ischaemic heart disease', 'Chronic obstructive pulmonary disease',
                'Stroke', 'Diarrhoeal diseases', 'Lower respiratory infections', 'Tuberculosis',
                'Diabetes mellitus', 'Cirrhosis of the liver', 'Falls'
            ],
            'Deaths': [221, 111, 70, 53, 34, 28, 25, 23, 19, 17]
        }

        df = pd.DataFrame(data)
        st.title("Leading Causes of Death")
        fig = px.bar(
            df,
            x='Cause',
            y='Deaths',
            title="Leading Causes of Death",
            color_discrete_sequence=['blue']
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title_text='Cause',
                gridcolor='lightgray',
                gridwidth=1,
                tickfont=dict(color='yellow',size=44) 
            ),
            yaxis=dict(
                title_text='Deaths',
                gridcolor='lightgray',
                gridwidth=1
            ),
            font=dict(
                color='black'
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_category == "HEALTHIER POPULATION":
        st.title("THE NUMBER OF PEOPLE EXPECTED TO LIVE HEALTHIER LIFE IN 2025")
        data = pd.DataFrame({
            'year': [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            'value': [0, 180, 280, 375, 475, 560, 620, 677]
        })
        confidence_interval = 80 
        data['upper_bound'] = data['value'] + confidence_interval
        data['lower_bound'] = data['value'] - confidence_interval
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data['year'].tolist() + data['year'].tolist()[::-1],
                y=data['upper_bound'].tolist() + data['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(65,105,225,0.1)', 
                line=dict(color='rgba(65,105,225,0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data['year'],
                y=data['value'],
                mode='lines+markers',
                name='Projection',
                line=dict(color='rgb(65,105,225)', width=2),  # Royal blue
                marker=dict(size=8, color='rgb(65,105,225)'),
                hovertemplate='Year: %{x}<br>Value: %{y}m<extra></extra>'
            )
        )
        fig.add_annotation(
            x=2025,
            y=677,
            text="677m",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=0,
            font=dict(color='rgb(65,105,225)', size=14),
            xanchor='left'
        )

        fig.update_layout(
            title={
                'text': 'Population Projection',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='Year',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[2018, 2025],
                dtick=1,
                showline=True,
                linecolor='rgba(0,0,0,0.2)'
            ),
            yaxis=dict(
                title='m',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[0, 800],
                tickmode='linear',
                tick0=0,
                dtick=200,  
                showline=True,
                linecolor='rgba(0,0,0,0.2)'
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            showlegend=False,  
            width=800,
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        st.plotly_chart(fig, use_container_width=True)
        
            
        

    elif selected_category == "ALCOHOL":
        st.title("PEOPLE CONSUMING  ALCOHOL")
        years = list(range(2000, 2021))
        np.random.seed(42) 
        data = {
            'Years': years,
            'Area1': np.linspace(0.5, 1, len(years)) + np.random.normal(0, 0.1, len(years)),
            'Area2': np.linspace(1, 1.5, len(years)) + np.random.normal(0, 0.1, len(years)),
            'Area3': np.linspace(1.5, 2, len(years)) + np.random.normal(0, 0.1, len(years)),
            'Area4': np.linspace(2, 2.5, len(years)) + np.random.normal(0, 0.1, len(years))
        }

        df = pd.DataFrame(data)
        df['Total'] = df[['Area1', 'Area2', 'Area3', 'Area4']].sum(axis=1)
        fig = go.Figure()
        areas = ['Area1', 'Area2', 'Area3', 'Area4']
        colors = ['rgba(230, 240, 230, 0.8)', 'rgba(200, 220, 200, 0.8)', 
                'rgba(170, 200, 170, 0.8)', 'rgba(140, 180, 140, 0.8)']

        for i, area in enumerate(areas):
            fig.add_trace(
                go.Scatter(
                    x=df['Years'],
                    y=df[area],
                    name=area,
                    mode='none',
                    stackgroup='one',
                    fillcolor=colors[i],
                    line=dict(width=0)
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df['Years'],
                y=df['Total'],
                name='Total',
                mode='lines+markers',
                line=dict(color='darkgreen', width=2),
                marker=dict(size=6, color='darkgreen')
            )
        )
        fig.update_layout(
            title=None,
            showlegend=True,
            plot_bgcolor='white',
            width=800,
            height=500,
            yaxis=dict(
                gridcolor='lightgray',
                gridwidth=1,
                dtick=5,
                range=[0, 15]
            ),
            xaxis=dict(
                gridcolor='lightgray',
                gridwidth=1,
                dtick=5
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        

        
    elif selected_category == "TOBACCO CONSUMERS AGED 15 AND OLDER":
        st.title("Tobacco consumers aged 15 and older upto 2030")
        years = list(range(2000, 2031, 5))
        data = pd.DataFrame({
            'year': list(range(2000, 2031)),
            'value': np.linspace(55, 19, 31) 
        })
        ci_1_width = 10  
        ci_2_width = 20  
        
        data['ci_1_upper'] = data['value'] + ci_1_width
        data['ci_1_lower'] = data['value'] - ci_1_width
        data['ci_2_upper'] = data['value'] + ci_2_width
        data['ci_2_lower'] = data['value'] - ci_2_width
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data['year'].tolist() + data['year'].tolist()[::-1],
                y=data['ci_2_upper'].tolist() + data['ci_2_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(200,200,200,0.3)',
                line=dict(color='rgba(200,200,200,0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data['year'].tolist() + data['year'].tolist()[::-1],
                y=data['ci_1_upper'].tolist() + data['ci_1_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,128,0,0.2)',
                line=dict(color='rgba(0,128,0,0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data['year'],
                y=data['value'],
                mode='lines+markers',
                name='Total',
                line=dict(color='darkgreen', width=2),
                marker=dict(size=8, color='darkgreen'),
                hovertemplate='Year: %{x}<br>Value: %{y:.1f}%<extra></extra>'
            )
        )
        fig.add_annotation(
            x=2010,
            y=40,
            text="40%",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=0,
            font=dict(color='darkgreen', size=14)
        )

        fig.add_annotation(
            x=2030,
            y=19,
            text="Total",
            showarrow=False,
            font=dict(color='darkgreen', size=14),
            xanchor='left',
            xshift=5
        )
        fig.update_layout(
            xaxis=dict(
                title='Year',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[2000, 2030],
                dtick=5, 
                showline=True,
                linecolor='rgba(0,0,0,0.2)'
            ),
            yaxis=dict(
                title='Percentage',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20,  
                tickformat='%',
                showline=True,
                linecolor='rgba(0,0,0,0.2)'
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            showlegend=False,
            width=800,
            height=500,
            margin=dict(l=50, r=50, t=30, b=50)
        )
        for year in data['year']:
            value = data.loc[data['year'] == year, 'value'].iloc[0]
            ci_1_upper = data.loc[data['year'] == year, 'ci_1_upper'].iloc[0]
            ci_1_lower = data.loc[data['year'] == year, 'ci_1_lower'].iloc[0]
            ci_2_upper = data.loc[data['year'] == year, 'ci_2_upper'].iloc[0]
            ci_2_lower = data.loc[data['year'] == year, 'ci_2_lower'].iloc[0]
            
            hover_text = (
                f'Year: {year}<br>'
                f'Value: {value:.1f}%<br>'
                f'Inner CI: {ci_1_lower:.1f}% - {ci_1_upper:.1f}%<br>'
                f'Outer CI: {ci_2_lower:.1f}% - {ci_2_upper:.1f}%'
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[year],
                    y=[value],
                    mode='markers',
                    marker=dict(opacity=0),
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=False
                )
            )
        st.plotly_chart(fig, use_container_width=True)

    
    elif selected_category =="POPULATION TREND":
        st.title("The population growth rate of India as of 2023 is 0.89 %")
        pass

    elif selected_category == "UNIVERSAL HEALTH COVERAGE":
        st.title("Health projection coverage graph")
        data = pd.DataFrame({
            'year': [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            'value': [0, 35, 20, 22, 92.9, 130, 155, 175]
        })

        confidence_interval = 25
        data['upper_bound'] = data['value'] + confidence_interval
        data['lower_bound'] = data['value'] - confidence_interval

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data['year'].tolist() + data['year'].tolist()[::-1],
                y=data['upper_bound'].tolist() + data['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,107,107,0.2)',
                line=dict(color='rgba(255,107,107,0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data['year'][:5],
                y=data['value'][:5],
                mode='lines+markers',
                name='Historical',
                line=dict(color='rgb(255,107,107)', width=2),
                hovertemplate='Year: %{x}<br>Value: %{y}m<extra></extra>'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data['year'][4:],
                y=data['value'][4:],
                mode='lines+markers',
                name='Projected',
                line=dict(color='rgb(255,107,107)', width=2, dash='dash'),
                hovertemplate='Year: %{x}<br>Value: %{y}m<extra></extra>'
            )
        )
        fig.add_annotation(
            x=2022,
            y=92.9,
            text="92.9m<br>India",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        fig.update_layout(
            title={
                'text': 'Population Projection',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='Year',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[2018, 2025]
            ),
            yaxis=dict(
                title='m',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[0, 250],
                tickmode='linear',
                tick0=0,
                dtick=50
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig, use_container_width=True)
   
        pass

    elif selected_category == "HEALTH EMERGENCIES PROJECTION ":
        st.title("Health emergencies projection")
    
        data = pd.DataFrame({
            'year': [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            'value': [0, 100, 50, 120, 240, 270, 310, 340]
        })

        confidence_interval = 40  
        data['upper_bound'] = data['value'] + confidence_interval
        data['lower_bound'] = data['value'] - confidence_interval
    
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data['year'].tolist() + data['year'].tolist()[::-1],
                y=data['upper_bound'].tolist() + data['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(147,112,219,0.2)',  
                line=dict(color='rgba(147,112,219,0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
  
        fig.add_trace(
            go.Scatter(
                x=data['year'][:5],
                y=data['value'][:5],
                mode='lines+markers',
                name='Historical',
                line=dict(color='rgb(147,112,219)', width=2), 
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Value: %{y}m<extra></extra>'
            )
        )
 
        fig.add_trace(
            go.Scatter(
                x=data['year'][4:],
                y=data['value'][4:],
                mode='lines+markers',
                name='Projected',
                line=dict(color='rgb(147,112,219)', width=2, dash='dash'),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Value: %{y}m<extra></extra>'
            )
        )

        fig.add_annotation(
            x=2022,
            y=240,
            text="240m<br>India",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color='rgb(147,112,219)', size=14)
        )

        fig.update_layout(
            title={
                'text': 'Population Projection',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='Year',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[2018, 2025],
                dtick=1  # Show every year
            ),
            yaxis=dict(
                title='m',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                range=[0, 400],
                tickmode='linear',
                tick0=0,
                dtick=100  # 100m intervals
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            showlegend=True,
            width=800,
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)

        pass

    elif selected_category == "Diseases":
        if selected_subcategory=="covid":
            dates = pd.date_range(start="2021-01-01", periods=365)
            new_cases = np.abs(np.random.normal(loc=100000, scale=50000, size=365))
            df = pd.DataFrame({"Date": dates, "New Cases": new_cases})
            fig = px.line(df, x="Date", y="New Cases", title="COVID-19 New Cases", 
                        labels={"New Cases": "New Cases"}, template="plotly_dark")
            fig.update_traces(hovertemplate='<b>Date</b>: %{x}<br><b>New Cases</b>: %{y:.0f}', line=dict(width=4))
            fig.update_layout(hoverlabel=dict(font_size=16))
            fig.show()
            plt.figure(figsize=(10, 6))
            plt.plot(df['Date'], df['New Cases'], color='blue', linewidth=2)
            plt.title("COVID-19 New Cases")
            plt.xlabel("Date")
            plt.ylabel("New Cases")
            plt.grid(True)
            plt.show()

        pass
