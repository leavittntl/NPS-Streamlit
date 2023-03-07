import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import r2_score 

# load & inspect dataset
data_table = "NPS_Species_Richness_Lat_Lon.csv"
df = pd.read_csv(data_table, index_col=0)





#Title
st.title('Exploring the use of GPT-3 as a data scraper, using the National Park Service (NPS) "Species Richness" report')
st.caption('By Nathaniel Leavitt (https://www.linkedin.com/in/natentl) and GPT-3')


#Overview
st.subheader('Overview')
st.write('This session examines the species richness of National Park Service parks in New England (with a total of 13 parks, determined by GPT-3). It provides a breakdown of the number of mammals, reptiles, amphibians, birds, fish, insects, mollusks, crustaceans, vascular plants, non-vascular plants, and fungi in each park, as of 2021. GPT-3 believes the data shows that the parks in New England have diverse and abundant wildlife, with a wide variety of species. And that, it also provides insight into the importance of preserving and protecting these parks, and their ecosystems, for future generations.')
st.write('The reason for using the NPS data set is for how it makes its Species Richness data available to the user. Currently, in order to view and download the required data for each park the site\'s user must first search and select one park from a "total possible" drop-down menu, select all of the species to include from a secondairy drop-down menu, click the "View Report" button, select an export option, then *répéter* for the other 12 parks. This does not include the time it takes to manually merge the data sets. Instead of going through this time consuming process, GPT-3 can be used to scrape the data for us within just a few seconds.' )

#Method
st.subheader('Method')
st.write('The data for this session was collected from the National Park Service website (www.nps.gov). Specifically 13 parks chosen by GPT-3, generated using the following prompt:')
st.write("""
        **Prompt:** \n
            What are the names of all National Park Service parks within New England? Group by State.
""")

st.write("""
        **GPT-3 Response:** \n
            Maine
            - Acadia National Park
            - Saint Croix Island International Historic Site
            
            New Hampshire
            - Saint-Gaudens National Historic Site
            - Appalachian National Scenic Trail
            - White Mountain National Forest

            Vermont
            - Marsh-Billings-Rockefeller National Historical Park

            Massachusetts
            - Adams National Historical Park
            - Boston African American National Historic Site
            - Boston Harbor Islands National Recreation Area
            - Cape Cod National Seashore
            - John F. Kennedy National Historic Site

            Rhode Island
            - Roger Williams National Memorial

            Connecticut
            - Weir Farm National Historic Site
""")


st.write("""To get the Species Richness data for each park, the follow prompt was used *"Get all "Species Richness" data from NPS.com, break out by specie type, within each park and display data in a table below"*, within GPT-3\'s playground environment. 
""")

st.write('The corrdinates for each park were also pulled using the prompt shown below, within GPT-3\'s playground environment:')
st.write("""
        **Prompt:** \n
            Please return the lat and lon for the following parks:\n
            Acadia National Park
            Saint Croix Island International Historic Site
            Saint-Gaudens National Historic Site
            Appalachian National Scenic Trail
            White Mountain National Forest
            Marsh-Billings-Rockefeller National Historical Park
            Adams National Historical Park
            Boston African American National Historic Site
            Boston Harbor Islands National Recreation Area
            Cape Cod National Seashore
            John F. Kennedy National Historic Site
            Roger Williams National Memorial
            Weir Farm National Historic Site
""")
st.write('After appending the lat and lon to the previouse data set, here is the resulting data frame:')
st.dataframe(df)
st.caption('*Note: This data is from 2021, based on the training data of GPT-3.')

st.write('Looks like two of the parks do not have any data collected ("Boston African American National Historic Site" and "John F. Kennedy National Historic Site "). These will be removed from the data set, using two pandas functions *.drop()* (to remove the empty rows) and *.reset_index()* (to reset the index column order).')
df = df.drop(labels=[7,10], axis=0)
df = df.reset_index(drop=True)
st.dataframe(df)

#Data
st.subheader('Data')

st.write("""The data frame should be restructed so that it is more useful, specifically when it comes to charting a 'stacked bar chart'. To do this, the *.melt()* function from the pandas libray is was implemented (some might know this as "reshaping from wide to long").""")
df2 = pd.melt(
    df, 
    id_vars=['Park Name','State','lat','lon'], 
    value_vars=['Mammals', 'Reptiles', 'Amphibians', 'Birds', 'Fish', 'Insects', 'Mollusks', 'Crustaceans', 'Vascular Plants', 'Non-Vascular Plants', 'Fungi'], 
    var_name ='Species', value_name ='RichnessCount'
    )

st.dataframe(df2)

#Data-First Stacked Bar Chart
c1 = alt.Chart(df2).mark_bar().encode(
    x='RichnessCount',
    y='Park Name',
    color='Species'
).properties(
    title='Species Richness per Park Name'
)


st.altair_chart(c1, use_container_width=True)


#Data-Second Stacked Bar Chart(Normalized)
c2 = alt.Chart(df2).mark_bar().encode(
    x=alt.X('RichnessCount', stack="normalize"),
    y='Park Name',
    color='Species'
).properties(
    title='Species Richness per Park Name (Normalized)'
)

st.altair_chart(c2, use_container_width=True)


st.write("""GPT-3 was then ask to look at the data set and return any staticailly significant findings, using the following promt:""")
st.write("""
        **Prompt:** \n
            Run explorative data analysis on this data set, return any statistically significance results
        """)
st.write("""
        **GPT-3 Response:** \n
            After running an exploratory data analysis on this data set, 
            we found that there is a statistically significant correlation between the number of mammals and the number of birds in the parks. 
            This suggests that the parks with higher mammal populations also tend to have higher bird populations. 
            Additionally, there is a statistically significant correlation between the number of insects and the number of birds, 
            suggesting that parks with higher insect populations also tend to have higher bird populations.
        """)

st.write("""As we can see, GPT-3 believes that there are at least two statistically significant correlations, the first being between mammals and bird, and another between insections and birds.
        To test this claim, here is a correlations matrix using pandas *.corr()* function (method = pearson): 
        """)

#Data-Corr
CorrValues = df.corr(method='pearson') 
st.write(CorrValues) 

st.write("""Looking at the two correlations values, based on GPT-3's claim, we can confirm that both claims are correct. Where mammals/birds = 0.77 and insects/birds = 0.97.
        That said, it looks like GPT-3 has missed several other correlations, some of which are much "stronger" than the mammals/birds correlation. 
        """)
st.write("""To close the loop on GPT-3's statistically significant claims, here are both correlations modeled in scatter plots (thanks to Vega-Altair's declarative statistical visualization library for Python https://altair-viz.github.io/):
        """)

#Data-Third Stacked Bar Chart(Normalized)
degree_list = [1, 3, 5]

base = alt.Chart(df).mark_circle(color="black").encode(
        alt.X("Mammals"), alt.Y("Birds")
).properties(
    title='Mammals vs. Birds'
)

polynomial_fit = [
    base.transform_regression(
        "Mammals", "Birds", method="poly", order=order, as_=["Mammals", str(order)]
    )
    .mark_line()
    .transform_fold([str(order)], as_=["degree", "Birds"])
    .encode(alt.Color("degree:N"))
    for order in degree_list
]

c3 = alt.layer(base, *polynomial_fit)


st.altair_chart(c3, use_container_width=True)



#Data-Fourth Stacked Bar Chart(Normalized)
base = alt.Chart(df).mark_circle(color="black").encode(
        alt.X("Insects"), alt.Y("Birds")
).properties(
    title='Insects vs. Birds'
)

polynomial_fit = [
    base.transform_regression(
        "Insects", "Birds", method="poly", order=order, as_=["Insects", str(order)]
    )
    .mark_line()
    .transform_fold([str(order)], as_=["degree", "Birds"])
    .encode(alt.Color("degree:N"))
    for order in degree_list
]

c4 = alt.layer(base, *polynomial_fit)


st.altair_chart(c4, use_container_width=True)

