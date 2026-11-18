import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_category_distribution(df: pd.DataFrame):
    """
    Bar chart of the top 15 product categories by complaint count.
    Returns a Plotly Figure, or None if 'Product' column is missing.
    """
    if 'Product' not in df.columns:
        return None

    counts = (
        df['Product']
        .value_counts()
        .nlargest(15)
        .reset_index()
    )
    counts.columns = ['Product', 'Count']

    fig = px.bar(
        counts,
        x='Count',
        y='Product',
        orientation='h',
        title='Top 15 Product Categories by Complaint Count',
        labels={'Count': 'Number of Complaints', 'Product': 'Product Category'},
        color='Count',
        color_continuous_scale='Blues',
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
    return fig


def get_monthly_trend(df: pd.DataFrame, date_col: str = 'Date received'):
    """
    Line chart of complaints per month.
    Returns a Plotly Figure, or None if the date column is missing.
    """
    if date_col not in df.columns:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    if df.empty:
        return None

    monthly = (
        df.set_index(date_col)
        .resample('ME')
        .size()
        .reset_index(name='Count')
    )
    monthly.columns = ['Month', 'Count']

    fig = px.line(
        monthly,
        x='Month',
        y='Count',
        title='Monthly Complaint Trend',
        labels={'Month': 'Month', 'Count': 'Number of Complaints'},
        markers=True,
    )
    fig.update_traces(line_color='steelblue')
    return fig


def get_company_analysis(df: pd.DataFrame, top_n: int = 15):
    """
    Horizontal bar chart of the top N companies by complaint count.
    Returns a Plotly Figure, or None if 'Company' column is missing.
    """
    if 'Company' not in df.columns:
        return None

    counts = (
        df['Company']
        .value_counts()
        .nlargest(top_n)
        .reset_index()
    )
    counts.columns = ['Company', 'Count']

    fig = px.bar(
        counts,
        x='Count',
        y='Company',
        orientation='h',
        title=f'Top {top_n} Companies by Complaint Count',
        labels={'Count': 'Number of Complaints', 'Company': 'Company'},
        color='Count',
        color_continuous_scale='Oranges',
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
    return fig


def get_status_distribution(df: pd.DataFrame):
    """
    Pie chart of 'Company response to consumer' distribution.
    Returns a Plotly Figure, or None if the column is missing.
    """
    col = 'Company response to consumer'
    if col not in df.columns:
        return None

    counts = df[col].value_counts().reset_index()
    counts.columns = ['Response', 'Count']

    fig = px.pie(
        counts,
        names='Response',
        values='Count',
        title='Company Response to Consumer Distribution',
        hole=0.3,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def get_complaint_heatmap(df: pd.DataFrame):
    """
    Heatmap of complaint counts: top-10 products vs top-10 companies.
    Returns a Plotly Figure, or None if required columns are missing.
    """
    if 'Product' not in df.columns or 'Company' not in df.columns:
        return None

    top_products = df['Product'].value_counts().nlargest(10).index.tolist()
    top_companies = df['Company'].value_counts().nlargest(10).index.tolist()

    subset = df[df['Product'].isin(top_products) & df['Company'].isin(top_companies)]

    if subset.empty:
        return None

    pivot = (
        subset.groupby(['Product', 'Company'])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure column order matches top_companies
    pivot = pivot.reindex(columns=[c for c in top_companies if c in pivot.columns], fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values.tolist(),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale='YlOrRd',
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title='Complaint Heatmap: Top 10 Products vs Top 10 Companies',
        xaxis_title='Company',
        yaxis_title='Product',
        xaxis_tickangle=-45,
    )
    return fig


def get_state_distribution(df: pd.DataFrame):
    """
    Choropleth map of complaints by US state.
    Returns a Plotly Figure, or None if 'State' column is missing.
    """
    if 'State' not in df.columns:
        return None

    counts = df['State'].value_counts().reset_index()
    counts.columns = ['State', 'Count']
    # Keep only 2-letter US state codes
    counts = counts[counts['State'].str.match(r'^[A-Z]{2}$', na=False)]

    if counts.empty:
        return None

    fig = px.choropleth(
        counts,
        locations='State',
        locationmode='USA-states',
        color='Count',
        scope='usa',
        title='Complaint Distribution by US State',
        labels={'Count': 'Number of Complaints'},
        color_continuous_scale='Reds',
    )
    return fig


def get_yearly_trend(df: pd.DataFrame):
    """
    Bar chart of complaints per year (derived from 'Date received').
    Returns a Plotly Figure, or None if 'Date received' column is missing.
    """
    date_col = 'Date received'
    if date_col not in df.columns:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    if df.empty:
        return None

    df['Year'] = df[date_col].dt.year
    yearly = df['Year'].value_counts().sort_index().reset_index()
    yearly.columns = ['Year', 'Count']

    fig = px.bar(
        yearly,
        x='Year',
        y='Count',
        title='Yearly Complaint Trend',
        labels={'Year': 'Year', 'Count': 'Number of Complaints'},
        color='Count',
        color_continuous_scale='Teal',
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig
