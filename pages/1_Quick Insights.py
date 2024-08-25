import altair as alt
import pandas as pd
import streamlit as st



#Page config
st.set_page_config(layout="wide",page_title="Quick Insights")




# Load and cache data
@st.cache_data
def load_and_prepare_data(filename):
    # Only load necessary columns
    data = pd.read_csv(filename)
    data['date_int'] = data['Id_date_agr'].astype(int)
    data['Id_date_agr'] = pd.to_datetime(data['Id_date_agr'], format='%Y%m')
    
    return data

df = load_and_prepare_data('../datasets/PNL_GL_EXTRACTION-19-08-2024_aggregated_per_month.csv')

selection_columns = ['Code journal', 

       'Rubrique_Altice', 'Société_IC', 'Filter_Type',
       'Rubrique_Altice_Réallo',

                  'Compte',
                   'Rubrique REP N1',
	'Rubrique REP N2',
	'Rubrique REP N3',
	'Rubrique REP N4',
	'Rubrique REP N5',
                   	'Rubrique N1',
	'Rubrique N2',
	'Rubrique N3',
	'Rubrique N4',
	'Rubrique N5',

	'Nom de section',

	'Nom site',

	'Nom société',

	'Client',

	'Projet',

	'Zone'                   
                   ]


# Define options for the dropdown lists

options = {f"{selection_columns[i]}": df[selection_columns[i]].unique().tolist() for i in range(len(selection_columns))}

user_selection = {
    #'range_date' : [df.date_int.min(), df.date_int.max()]
}


# Cache the filtered data to avoid recomputation
@st.cache_data
# Function to filter data based on user selection
def filter_data(data, user_selection):
    
    #date filters
    min_date_selected = user_selection["range_date"][0]
    max_date_selected = user_selection["range_date"][1]
    
    #Remove the date part
    user_selection = dict(list(user_selection.items())[1:])

    mask = pd.Series([True] * len(data), index=data.index)

    # Apply filtering only for non-empty lists in filter_dict
    for col, values in user_selection.items():
        if values:  # Apply filtering only if the values list is not empty
            mask &= data[col].isin(values)

    # Apply the mask to filter the DataFrame
    filtered_data = data[mask]

    #Apply date filter
    filtered_data = filtered_data[filtered_data.date_int.between(min_date_selected, max_date_selected)]

    return filtered_data




st.title("Quick Insights with Filters")


# Filters part, in sidebar
with st.sidebar.expander("Appliquer des filtres spécifiques"):

        selected_date_range = st.slider(
        "Select Date Range", 
        min_value=int(df['date_int'].min()), 
        max_value=int(df['date_int'].max()), 
        value=(int(df['date_int'].min()), int(df['date_int'].max()))
    )
        
        user_selection = {
            'range_date' : selected_date_range
        }
        
        
        for i, (col_name, values) in enumerate(options.items()):
            
                selected_values = st.multiselect(col_name, options=sorted(values, key=str), placeholder='Choisir les valeurs')
                user_selection[col_name] = selected_values
                

# Apply filters                
filtered_data = filter_data(df, user_selection)


# Montant trend by site and date
with st.expander("Show Stacked Area Chart for Montant(€) by Site"):
    st.subheader("Montant(€) by Site Over Time")
    budget_by_site_time = filtered_data.groupby(['Id_date_agr', 'Nom site'])['Montant(€)'].sum().reset_index()
    area_chart = alt.Chart(budget_by_site_time).mark_area().encode(
        alt.X('Id_date_agr:T', axis=alt.Axis(title='Date')),
         y=alt.Y('Montant(€):Q', axis=alt.Axis(title='Montant(€)')),  # Renaming y-axis
    color=alt.Color('Nom site:N', legend=alt.Legend(title='Site Name')),  # Renaming legend
        tooltip=['Id_date_agr:T', 'Nom site:N', 'Montant(€):Q']
    ).interactive()
    st.altair_chart(area_chart, use_container_width=True)


#barchart to display Montant(€) per Rubrique
with st.expander("Show Total Montant(€) by Rubrique"):
    st.subheader("Total Montant by Rubrique")
    budget_by_rub = filtered_data.groupby('Rubrique N5')['Montant(€)'].sum().reset_index()
    bar_chart_rub = alt.Chart(budget_by_rub).mark_bar().encode(
        x=alt.X('Montant(€):Q', title="Total Montant"),
        y=alt.Y('Rubrique N5:N', sort='-x', title="Rubrique N5"),
        tooltip=['Rubrique N5:N', 'Montant(€):Q']
    ).interactive()
    st.altair_chart(bar_chart_rub, use_container_width=True)

#recchart to display Montant(€) per Rubrique & Month
with st.expander("Show Heatmap for Montant(€) by Month and Rubrique"):
    st.subheader("Montant(€) by Month and Rubrique")
    heatmap_data = filtered_data.groupby(['Id_date_agr', 'Rubrique N5'])['Montant(€)'].sum().reset_index()
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x='Id_date_agr:T',
        y='Rubrique N5:N',
        color='Montant(€):Q',
        tooltip=['Id_date_agr:T', 'Rubrique N5:N', 'Montant(€):Q']
    ).interactive()
    st.altair_chart(heatmap, use_container_width=True)

# Most frequent Section
with st.expander("10 Most Frequent Section"):
    st.subheader("Most Frequent Section")
    rubrique_counts = filtered_data['Nom de section'].value_counts().reset_index().head(10)
    rubrique_counts.columns = ['Nom de section', 'count']
    pie_chart = alt.Chart(rubrique_counts).mark_arc().encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field="Nom de section", type="nominal"),
        tooltip=['Nom de section:N', 'count:Q']
    )
    st.altair_chart(pie_chart, use_container_width=True)

#barchart to display Montant(€) per Rubrique
with st.expander("Show Total Montant(€) by Section Name"):
    st.subheader("Total Budget by Section Name")
    budget_by_rub = filtered_data.groupby('Nom de section')['Montant(€)'].sum().reset_index()
    bar_chart_rub = alt.Chart(budget_by_rub).mark_bar().encode(
        x=alt.X('Montant(€):Q', title="Total Montant"),
        y=alt.Y('Nom de section:N', sort='x', title="Nom de section"),
        tooltip=['Nom de section:N', 'Montant(€):Q']
    ).interactive()
    st.altair_chart(bar_chart_rub, use_container_width=True)

# Bar chart Montant per Month
with st.expander("Show Monthly Spending"):
    st.subheader("Monthly Spending")
    monthly_spending = filtered_data.groupby('Id_date_agr')['Montant(€)'].sum().reset_index()
    bar_chart_month = alt.Chart(monthly_spending).mark_bar().encode(
        x='Id_date_agr:T',
        y='Montant(€):Q',
        tooltip=['Id_date_agr:T', 'Montant(€):Q']
    ).interactive()
    st.altair_chart(bar_chart_month, use_container_width=True)



with st.expander("Correlation between Montant for each Filter Type"):
    # Pivot the dataframe to get filter_type as columns
    
    filtered_data = filtered_data.groupby(['Id_date_agr','Filter_Type'], as_index=False).agg({'Montant(€)' : 'sum'})
    
    pivot_df = filtered_data.pivot(index='Id_date_agr', columns='Filter_Type', values='Montant(€)')

    # Calculate the correlation matrix between the different filter_type columns
    correlation_matrix = pivot_df.corr()

    # Calculate the correlation matrix between the different filter_type columns
    correlation_matrix = pivot_df.corr()
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Plot the correlation matrix

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Budgets by Filter Type')

    # Display the plot in Streamlit
    st.pyplot(fig)