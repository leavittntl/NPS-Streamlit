import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# load & inspect dataset
data_table = "NPS_Species_Richness_V2.csv"
df = pd.read_csv(data_table, index_col=0)





#Title
st.title('Exploring the use of GPT-3 as a data scraper, using the National Park Service (NPS) "Species Richness" report')
st.caption('By: Nathaniel Leavitt (https://www.linkedin.com/in/natentl) and GPT-3')
st.caption('Posted on: 03/15/2023')
st.caption('Model: text-davinci-003; Temperature: 0.7')

#Intro
st.subheader('Intro:')
st.write("""
    This session explores the usefulness of GPT-3, in regards to data collection across online resources. 
    The criteria for evaluation will be broken out into three categories. 
    First, we will be looking at the amount of time it takes for GPT-3 to collect a full data set.
    Next, the utility of the data set will be explored via table renderings and chart visualizations.
    Lastly, GPT-3’s understanding of what it has accomplished and collected will be tested by asking it to summarize an “overview”, “method”, and “analysis” section(s).
""")

#Collection
st.subheader('Data Set Constrution:')
st.write("""

    Establishing a foundational understanding of where the data is coming from is the first point of interest, 
    in this case a group of national parks within New England (which was arbitrarily chosen as the region of interest). 
    GPT-3 was asked to return a list of parks under the National Park Service: \n
        Prompt: \n
            What are the names of all National Park Service parks within New England? Group by State.

        GPT-3 Response: \n
            Maine
                -Acadia National Park
                -Saint Croix Island International Historic Site

            New Hampshire 
                -Saint-Gaudens National Historic Site
                -Appalachian National Scenic Trail
                -White Mountain National Forest

            Vermont
                -Marsh-Billings-Rockefeller National Historical Park

            Massachusetts
                -Adams National Historical Park
                -Boston African American National Historic Site
                -Boston Harbor Islands National Recreation Area
                -Cape Cod National Seashore
                -John F. Kennedy National Historic Site

            Rhode Island
                -Roger Williams National Memorial

            Connecticut
                -Weir Farm National Historic Site
    
    The response shows a list of 13 parks, grouped by State. 
    The completion of this request was fulfilled within seconds and will define the scope of our data collection.
""")

st.write("""
    Now we want to start collecting the data. 
    We know that the data is accessible via the site “https://irma.nps.gov/NPSpecies/Reports/Systemwide/Species%20Richness”, 
    based on previous sleuthing of the National Park Service site (https://www.nps.gov/index.htm). 
    For the first test, while in the same playground session of GPT-3 (so that GPT-3 still has access to the list of parks generated from before), 
    the species richness data was requested:

        Prompt: \n
            Get all "Species Richness" data from NPS.com, 
            break out by specie type, within eack park and display data in a table below

        GPT-3 Response: \n
            Park Name | Mammals | Reptiles | Amphibians | Birds | Fish
            --- | --- | --- | --- | --- | --- 
            Acadia National Park | 22 | 8 | 8 | 283 | 8
            Saint Croix Island International Historic Site | 0 | 0 | 0 | 24 | 0
            Saint-Gaudens National Historic Site | 0 | 0 | 0 | 28 | 0
            Appalachian National Scenic Trail | 32 | 12 | 11 | 464 | 11
            White Mountain National Forest | 35 | 3 | 3 | 140 | 3
            Marsh-Billings-Rockefeller National Historical Park | 20 | 7 | 12 | 114 | 0
            Adams National Historical Park | 0 | 0 | 0 | 32 | 0
            Boston African American National Historic Site | 0 | 0 | 0 | 0 | 0
            Boston Harbor Islands National Recreation Area | 6 | 3 | 3 | 110 | 2
            Cape Cod National Seashore | 10 | 5 | 6 | 202 | 4
            John F. Kennedy National Historic Site | 0 | 0 | 0 | 0 | 0
            Roger Williams National Memorial | 0 | 0 | 0 | 0 | 0
            Weir Farm National Historic Site | 0 | 0 | 0 | 0 | 0

    The returned data looks great, but let’s increase the range of species captured.

        Prompt: \n 
            Try again, please add Insects, Mollusks, 
            Crustaceans, Vascular Plants, Non-Vascular Plants, Fungi

        GPT-3 Response: \n
            Park Name | Mammals | Reptiles | Amphibians | Birds | Fish | Insects | Mollusks | Crustaceans | Vascular Plants | Non-Vascular Plants | Fungi
            --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
            Acadia National Park | 22 | 8 | 8 | 283 | 8 | 459 | 4 | 5 | 416 | 70 | 48
            Saint Croix Island International Historic Site | 0 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 26 | 3 | 5
            Saint-Gaudens National Historic Site | 0 | 0 | 0 | 28 | 0 | 0 | 0 | 0 | 25 | 7 | 5
            Appalachian National Scenic Trail | 32 | 12 | 11 | 464 | 11 | 562 | 2 | 3 | 437 | 111 | 83
            White Mountain National Forest | 35 | 3 | 3 | 140 | 3 | 163 | 0 | 0 | 568 | 91 | 39
            Marsh-Billings-Rockefeller National Historical Park | 20 | 7 | 12 | 114 | 0 | 86 | 0 | 0 | 539 | 112 | 35
            Adams National Historical Park | 0 | 0 | 0 | 32 | 0 | 0 | 0 | 0 | 53 | 4 | 8
            Boston African American National Historic Site | 0 | 0 | 0 | 0 | 0 | 0  | 0 | 0 | 0 | 0 | 0
            Boston Harbor Islands National Recreation Area | 6 | 3 | 3 | 110 | 2 | 174 | 2 | 0 | 367 | 50 | 11
            Cape Cod National Seashore | 10 | 5 | 6 | 202 | 4 | 170 | 5 | 1 | 447 | 67 | 24
            John F. Kennedy National Historic Site | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
            Roger Williams National Memorial | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 4 | 4
            Weir Farm National Historic Site | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 88 | 10 | 8

""")
st.write("""
    This looks great! All of this data was scraped within a matter of just a few minutes. 
    If we contrast this with the amount of time it would take for a user to manually gather this data, 
    where the process would look like something along the lines as follows:

    1.	Select one park from a "total possible" drop-down menu \n
    2.	Select all of the species to include from a secondary drop-down menu \n
    3.	Click the "View Report" button \n
    4.	Select an export option \n
    5.	Répéter for the other 12 parks… \n

    This does not include the time it takes to manually merge the data sets.
""")

st.write("""
    Which leads us to the last point of interest within this section, 
    exporting/saving this data into a useful format. Luckly, 
    this can be accomplish quickly by coping and pasting the out from GPT-3 into an excel spreadsheet and reformatting the data, 
    using the “Text to Columns” data tool. Unfortunately, at this time, 
    the GPT-3 playground does not allow for the exportation of data tables generated by GPT-3.
""")

#Utility
st.subheader('Explore Utility:')
st.write("""
    Much of the data exploration for this section will be carried out using “pandas” and “Vega-Altair”, 
    both of which are libraries for the programing language Python. Please feel free to look more into these libraries, 
    but for the purposes of this GPT-3 “experiment”, most of this section will be brief and surface level.
""")
st.write("""
    Using pandas, we can import the dataset from the previous section and display it as a table (shown below):
""")
st.dataframe(df)
st.caption('*Note: This data is from 2021, based on the training data of GPT-3.')


st.write("""
    Looks like two of the parks do not have any data collected ("Boston African American National Historic Site" and "John F. Kennedy National Historic Site "). 
    These will be removed from the data set, using two pandas functions *.drop()* (to remove the empty rows) and *.reset_index()* (to reset the index column order).
""")
df = df.drop(labels=[7,10], axis=0)
df = df.reset_index(drop=True)
st.dataframe(df)

st.write("""
    Let’s view this data as a stacked bar chart (the data frame was reshaped using the pandas .melt() function):
""")

df2 = pd.melt(
    df, 
    id_vars=['Park Name','State'], 
    value_vars=['Mammals', 'Reptiles', 'Amphibians', 'Birds', 'Fish', 'Insects', 'Mollusks', 'Crustaceans', 'Vascular Plants', 'Non-Vascular Plants', 'Fungi'], 
    var_name ='Species', value_name ='RichnessCount'
    )

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


st.write("""
    Finally, we can quickly look at a correlation matrix using the pandas .corr() function:
""")
CorrValues = df.corr(method='pearson') 
st.write(CorrValues)

st.write("""
    Everything in section is designed to show the usefulness of the data set gathered by GPT-3. 
    Based on only needing just a few lines of python and Vega-Altair code to visualize and analyze the data, 
    this data set is very easy to use and understand.
        """)

#GPT-3 Explore
st.subheader('GPT-3 Explore:')
st.write("""
    After proving that the data collect by GPT-3 is easy to use, let's explore GPT-3's understanding of the data.

    Ask GPT-3 to write an “overview” section:

        Prompt: \n 
            Write an "overview" for this session.

        GPT-3 Response: \n
            This session examines the species richness of National Park Service 
            parks in New England. It provides a breakdown of the number of mammals, 
            reptiles, amphibians, birds, fish, insects, mollusks, crustaceans, 
            vascular plants, non-vascular plants, and fungi in each park, as of 2021. 
            The data shows that the parks in New England have diverse and abundant 
            wildlife, with a wide variety of species. It also provides insight into 
            the importance of preserving and protecting these parks, 
            and their ecosystems, for future generations.

""")
st.write("""
    Ask GPT-3 to write a “method” section:

        Prompt: \n 
            Write a "method" section for this session.

        GPT-3 Response: \n
            The data for this session was collected from 
            the National Park Service website (www.nps.gov). 
            Species richness data was broken out by species type, 
            within each park, and displayed in a table. 
            The data was then further broken out into vascular 
            and non-vascular plants, and fungi, to provide a more 
            detailed overview of species richness.            


""")

st.write("""
    Ask GPT-3 to explore data:

        Prompt: \n 
            Run explorative data analysis on this data set, 
            return any statistically significance results

        GPT-3 Response: \n
            After running an exploratory data analysis on 
            this data set, we found that there is a statistically 
            significant correlation between the number of mammals 
            and the number of birds in the parks. 
            This suggests that the parks with higher mammal populations 
            also tend to have higher bird populations. 
            Additionally, there is a statistically significant 
            correlation between the number of insects and the number 
            of birds, suggesting that parks with higher insect 
            populations also tend to have higher bird populations.            


""")

#Conclusion
st.subheader('Closing Thoughts:')
st.write("""
    Overall, the use of GPT-3 for data collection, aggregation, 
    and site scraping is very easy to use and can save a significant amount time. 
    Even though its assessment of the findings was very surface level, 
    it still can be used to present possible avenues of further exploration.
    
    Somethings to note: \n
    1.	There were some variations within the names of parks returned, when re-testing the “” prompt. This might be mitigated by adjusting the “temperature” setting down to “0”, but further testing is required.
    2.	The data returned is outdated. This is due to GPT-3 being trained on a version of the internet cached backed in 2021. Unfortunately, this cannot be controlled by the user. That said, the release of GPT-4 might provide us with more up-to-date data.

""")
