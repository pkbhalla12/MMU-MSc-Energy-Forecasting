import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\Prashant Bhalla\Desktop\Prashant-Docs\MMU\MSc Project\new"
TRAIN_PATH = os.path.join(BASE, "train.csv")
TEST_PATH = os.path.join(BASE, "test.csv")
RESULTS_PATH = os.path.join(BASE, "Output", "model_results_daily_fair.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=['datetime'])
    test = pd.read_csv(TEST_PATH, parse_dates=['datetime'])
    results = pd.read_csv(RESULTS_PATH)
    return train, test, results

train, test, results = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ UK National Electricity Demand Forecasting")
st.markdown("**MSc Data Science Dissertation | Prashant Kumar Bhalla | Manchester Metropolitan University**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dataset Overview",
    "🔍 Exploratory Data Analysis",
    "📈 Model Comparison",
    "🔮 Forecast Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.header("Dataset Overview")
    st.markdown("**UK National Electricity Consumption 2009-2024** (Vidalrod, 2024)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", "279,264")
    with col2:
        st.metric("Time Period", "2009-2024")
    with col3:
        st.metric("Granularity", "Half-hourly")
    with col4:
        st.metric("Features", "37 (engineered)")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Set Statistics")
        stats = train['nd'].describe().round(2)
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean (MW)', 'Std Dev (MW)', 'Min (MW)',
                         '25th Percentile (MW)', 'Median (MW)',
                         '75th Percentile (MW)', 'Max (MW)'],
            'Value': [
                f"{len(train):,}",
                f"{train['nd'].mean():,.0f}",
                f"{train['nd'].std():,.0f}",
                f"{train['nd'].min():,.0f}",
                f"{train['nd'].quantile(0.25):,.0f}",
                f"{train['nd'].median():,.0f}",
                f"{train['nd'].quantile(0.75):,.0f}",
                f"{train['nd'].max():,.0f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Test Set Statistics")
        test_stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean (MW)', 'Std Dev (MW)', 'Min (MW)',
                         '25th Percentile (MW)', 'Median (MW)',
                         '75th Percentile (MW)', 'Max (MW)'],
            'Value': [
                f"{len(test):,}",
                f"{test['nd'].mean():,.0f}",
                f"{test['nd'].std():,.0f}",
                f"{test['nd'].min():,.0f}",
                f"{test['nd'].quantile(0.25):,.0f}",
                f"{test['nd'].median():,.0f}",
                f"{test['nd'].quantile(0.75):,.0f}",
                f"{test['nd'].max():,.0f}"
            ]
        })
        st.dataframe(test_stats_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature List")
    features_df = pd.DataFrame({
        'Feature': ['nd', 'settlement_period', 'hour', 'month', 'weekday',
                   'is_weekend', 'is_holiday', 'is_covid',
                   'embedded_wind_generation', 'embedded_solar_generation',
                   'lag_1', 'lag_2', 'lag_48', 'lag_336',
                   'rolling_mean_48', 'rolling_std_48', 'rolling_mean_336'],
        'Description': [
            'National Demand (MW) — target variable',
            'Half-hour slot within the day (1-48)',
            'Hour of day (0-23)',
            'Month of year (1-12)',
            'Day of week (0=Monday)',
            'Weekend indicator (1=Sat/Sun)',
            'Bank holiday indicator',
            'COVID-19 lockdown period indicator (Mar 2020-Jul 2021)',
            'Embedded wind generation (MW)',
            'Embedded solar generation (MW)',
            'Demand at previous half-hour',
            'Demand two half-hours prior',
            'Demand at same time yesterday',
            'Demand at same time last week',
            '24-hour rolling mean demand',
            '24-hour rolling std deviation',
            '1-week rolling mean demand'
        ],
        'Type': ['Target', 'Temporal', 'Temporal', 'Temporal', 'Temporal',
                'Temporal', 'Temporal', 'Temporal',
                'Exogenous', 'Exogenous',
                'Lag', 'Lag', 'Lag', 'Lag',
                'Rolling', 'Rolling', 'Rolling']
    })
    st.dataframe(features_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    # Full time series
    st.subheader("UK National Demand - Full Time Series (2009-2024)")

    sample_train = train.iloc[::48].copy()  # daily sample for speed
    sample_test = test.iloc[::48].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_train['datetime'], y=sample_train['nd'],
        name='Training (2009-2022)', line=dict(color='steelblue', width=0.8)
    ))
    fig.add_trace(go.Scatter(
        x=sample_test['datetime'], y=sample_test['nd'],
        name='Test (2023-2024)', line=dict(color='darkorange', width=0.8)
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="National Demand (MW)",
        height=400, hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # COVID highlight
    st.subheader("COVID-19 Impact on Electricity Demand")
    covid_start = pd.Timestamp('2020-03-23')
    covid_end = pd.Timestamp('2021-07-19')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sample_train['datetime'], y=sample_train['nd'],
        name='National Demand', line=dict(color='steelblue', width=0.8)
    ))
    fig2.add_vrect(
        x0=covid_start, x1=covid_end,
        fillcolor='red', opacity=0.15,
        layer='below', line_width=0,
        annotation_text='COVID-19 Period', annotation_position='top left'
    )
    fig2.update_layout(
        xaxis_title="Date", yaxis_title="National Demand (MW)",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    col1, col2 = st.columns(2)

    with col1:
        monthly = train.groupby('month')['nd'].mean().reset_index()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']
        monthly['month_name'] = monthly['month'].apply(lambda x: month_names[x-1])
        fig3 = px.bar(monthly, x='month_name', y='nd',
                     title='Average Demand by Month',
                     labels={'nd': 'Avg Demand (MW)', 'month_name': 'Month'},
                     color='nd', color_continuous_scale='Blues')
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        hourly = train.groupby('hour')['nd'].mean().reset_index()
        fig4 = px.line(hourly, x='hour', y='nd',
                      title='Average Demand by Hour of Day',
                      labels={'nd': 'Avg Demand (MW)', 'hour': 'Hour'})
        fig4.update_traces(line_color='steelblue')
        st.plotly_chart(fig4, use_container_width=True)

    # Yearly trend
    st.subheader("Long-Term Demand Trend")
    yearly = train.groupby(train['datetime'].dt.year)['nd'].mean().reset_index()
    yearly.columns = ['year', 'avg_demand']
    fig5 = px.line(yearly, x='year', y='avg_demand',
                  title='Average Annual National Demand (MW)',
                  labels={'avg_demand': 'Avg Demand (MW)', 'year': 'Year'},
                  markers=True)
    fig5.update_traces(line_color='steelblue', marker_color='darkorange')
    st.plotly_chart(fig5, use_container_width=True)

    st.info(f"Peak demand: {train['nd'].max():,.0f} MW (2010) → "
            f"Latest peak: {test['nd'].max():,.0f} MW (2023-2024). "
            f"Demand has fallen by approximately 27% over 15 years.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.header("Model Comparison")
    st.markdown("All models evaluated on daily-aggregated predictions for a fair, scale-consistent comparison.")

    # Results table
    st.subheader("Performance Summary (Daily-Aggregated)")
    results_display = results.copy()
    results_display = results_display.sort_values('MAPE').reset_index(drop=True)
    results_display.index = results_display.index + 1
    results_display.columns = ['Model', 'RMSE (MW)', 'MAE (MW)', 'MAPE (%)']
    st.dataframe(results_display, use_container_width=True)

    st.markdown("---")

    # MAPE bar chart
    col1, col2 = st.columns(2)

    with col1:
        fig6 = px.bar(
            results_display, x='Model', y='MAPE (%)',
            title='MAPE Comparison (%) — Lower is Better',
            color='MAPE (%)', color_continuous_scale='RdYlGn_r',
            text='MAPE (%)'
        )
        fig6.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig6.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        fig7 = px.bar(
            results_display, x='Model', y='RMSE (MW)',
            title='RMSE Comparison (MW) — Lower is Better',
            color='RMSE (MW)', color_continuous_scale='RdYlGn_r',
            text='RMSE (MW)'
        )
        fig7.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig7.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Findings")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**🏆 Best Model: XGBoost**\nMAPE 0.29% - strong short-term autocorrelation favours tree-based models with lag features")
    with col2:
        st.info("**⚡ Best Deep Learning: GRU**\nMAPE 0.59% - GRU outperforms LSTM, consistent with Alomari et al. (2023)")
    with col3:
        st.warning("**📉 Hybrid Underperforms**\nMAPE 11.94% - ARIMA's inability to capture seasonal variation limits the hybrid approach")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast Explorer":
    st.header("Forecast Explorer")
    st.markdown("Explore actual vs predicted demand patterns in the test set (2023-2024).")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.Timestamp("2023-01-01"))
    with col2:
        end_date = st.date_input("End date", value=pd.Timestamp("2023-01-31"))

    # Filter test data
    mask = (test['datetime'].dt.date >= start_date) & (test['datetime'].dt.date <= end_date)
    test_filtered = test[mask]

    if len(test_filtered) == 0:
        st.warning("No data for selected date range. Please select dates within 2023-2024.")
    else:
        st.subheader(f"Actual Demand — {start_date} to {end_date}")
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=test_filtered['datetime'],
            y=test_filtered['nd'],
            name='Actual Demand',
            line=dict(color='steelblue', width=1.5)
        ))
        fig8.update_layout(
            xaxis_title="Date", yaxis_title="National Demand (MW)",
            height=400, hovermode='x unified'
        )
        st.plotly_chart(fig8, use_container_width=True)

        # Daily stats for selected period
        st.subheader("Daily Statistics for Selected Period")
        daily_stats = test_filtered.groupby(test_filtered['datetime'].dt.date)['nd'].agg(
            ['mean', 'min', 'max', 'std']
        ).round(0).reset_index()
        daily_stats.columns = ['Date', 'Mean (MW)', 'Min (MW)', 'Max (MW)', 'Std Dev (MW)']
        st.dataframe(daily_stats, hide_index=True, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Demand", f"{test_filtered['nd'].mean():,.0f} MW")
        with col2:
            st.metric("Peak Demand", f"{test_filtered['nd'].max():,.0f} MW")
        with col3:
            st.metric("Minimum Demand", f"{test_filtered['nd'].min():,.0f} MW")

    st.markdown("---")
    st.subheader("Weekend vs Weekday Demand Pattern")
    weekend = test[test['is_weekend'] == 1].groupby('hour')['nd'].mean()
    weekday = test[test['is_weekend'] == 0].groupby('hour')['nd'].mean()

    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=weekday.index, y=weekday.values,
                             name='Weekday', line=dict(color='steelblue', width=2)))
    fig9.add_trace(go.Scatter(x=weekend.index, y=weekend.values,
                             name='Weekend', line=dict(color='darkorange', width=2)))
    fig9.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Average Demand (MW)",
        height=400, hovermode='x unified'
    )
    st.plotly_chart(fig9, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*MSc Data Science Dissertation - Prashant Kumar Bhalla (25948685) - Manchester Metropolitan University - 2026*")