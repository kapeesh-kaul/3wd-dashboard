from functools import partial
from shiny.express import ui, input
from shiny import render
from shiny.ui import page_navbar
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from shinywidgets import render_widget 
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import networkx as nx

papers = pd.read_csv("Research_Papers_Table_with_All_Additional_Entries.csv")

ui.page_opts(
    title="Three Way Decisions in Disaster Management",  
    page_fn=partial(page_navbar, id="page"),  
)


with ui.nav_panel("Overview"):
    with ui.layout_columns():
        with ui.value_box():
            'Number of Papers'
            len(papers)

        with ui.value_box():
            'Number of Unique Authors'
            authors = papers['Authors'].dropna().str.split(',').explode().str.strip().unique()
            len(authors)
        
        with ui.value_box():
            'Number of Conferences'
            conferences = papers['Journal/Conference'].dropna().unique()
            len(conferences)
    
    with ui.card():
        ui.card_header("An Overview of the Dataset")

        @render.data_frame
        def papers_table():
            # return render.DataGrid(papers.iloc[:, 1:].head())
            def truncate_string(s, length=30):
                return s if len(s) <= length else s[:length] + '...'

            papers_truncated = papers.copy()
            for col in papers_truncated.columns:
                if papers_truncated[col].dtype == 'object':
                    papers_truncated[col] = papers_truncated[col].apply(lambda x: truncate_string(x) if isinstance(x, str) else x)

            return render.DataGrid(papers_truncated.iloc[:, 1:].head())

with ui.nav_panel("Insights"):
    ui.HTML('''
        <div align = 'center'>
            <h4>Survey Statistics</h4>
        </div>
    ''')
    with ui.layout_columns():
        with ui.card():
            ui.card_header("Number of Papers per Year")
            @render_widget
            def plot_papers_per_year():
                papers_filtered = papers.dropna(subset=['Publication Year'])

                # Filter out non-numeric 'Publication Year' values
                papers_filtered = papers_filtered[papers_filtered['Publication Year'].apply(lambda x: x.isnumeric())]

                # Convert 'Publication Year' to integer
                papers_filtered['Publication Year'] = papers_filtered['Publication Year'].astype(int)

                # Plot the distribution of papers over the years
                papers_per_year = papers_filtered['Publication Year'].value_counts().sort_index()
                fig = px.line(x=papers_per_year.index, y=papers_per_year.values, labels={'x': 'Year', 'y': 'Number of Papers'})
                fig.update_layout(template='simple_white')
                return fig

        with ui.card():
            ui.card_header("Number of Papers per Conference")
            @render_widget
            def plot_papers_per_conference():
                papers_per_conference = papers['Journal/Conference'].value_counts().sort_values(ascending=True)
                fig = px.bar(x=papers_per_conference.values, y=papers_per_conference.index, orientation='h', labels={'x': 'Number of Papers', 'y': 'Conference'})
                fig.update_layout(template='simple_white')
                return fig
            
    ui.HTML('''
        <div align = 'center'>
            <h4>Sentiment Analysis</h4>
        </div>
    ''')
    with ui.layout_columns():    
        with ui.card():
            ui.card_header("Sentiment polarity of Abstracts")
            @render_widget
            def plot_abstract_polarity():
                def get_sentiment(text):
                    blob = TextBlob(text)
                    return blob.sentiment.polarity

                # Apply the function to the 'Abstract' column
                papers['Sentiment'] = papers['Abstract Summary'].dropna().apply(get_sentiment)

                # Plot the distribution of sentiment polarity
                fig = px.histogram(papers, x='Sentiment', nbins=20, labels={'x': 'Sentiment Polarity', 'y': 'Number of Papers'})
                fig.update_layout(template='simple_white')
                return fig
        with ui.card():
            ui.card_header("Sentiment polarity of 3-Way Feasibility")
            @render_widget
            def plot_3way_feasibility_polarity():
                def get_sentiment(text):
                    blob = TextBlob(text)
                    return blob.sentiment.polarity

                # Apply the function to the '3-Way Feasibility' column
                papers['Sentiment'] = papers['3-Way Feasibility'].dropna().apply(get_sentiment)

                # Plot the distribution of sentiment polarity
                fig = px.histogram(papers, x='Sentiment', nbins=20, labels={'x': 'Sentiment Polarity', 'y': 'Number of Papers'})
                fig.update_layout(template='simple_white')
                return fig
            
        with ui.card():
            ui.card_header("Sentiment polarity of Future Work")
            @render_widget
            def plot_future_work_polarity():
                def get_sentiment(text):
                    blob = TextBlob(text)
                    return blob.sentiment.polarity

                # Apply the function to the 'Future Work Suggestions' column
                papers['Sentiment'] = papers['Future Work Suggestions'].dropna().apply(get_sentiment)

                # Plot the distribution of sentiment polarity
                fig = px.histogram(papers, x='Sentiment', nbins=20, labels={'x': 'Sentiment Polarity', 'y': 'Number of Papers'})
                fig.update_layout(template='simple_white')
                return fig

with ui.nav_panel("WordClouds"):
    with ui.layout_columns():  
        with ui.card():  
            ui.card_header("Wordcloud for Keywords")
            @render.plot
            def wordcloud_keyword():
                
                # Combine all keywords into a single string
                keywords_text = ' '.join(papers['Keywords'].dropna())

                # Generate the word cloud
                wordcloud_keyword = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_keyword, interpolation='bilinear')
                ax.axis('off')

                return fig

        with ui.card():  
            ui.card_header("Wordcloud for Future Work")
            @render.plot
            def wordcloud_future_work():
                # papers = pd.read_csv("paper.csv")
                
                # Combine all keywords into a single string
                keywords_text = ' '.join(papers['Future Work Suggestions'].dropna())

                # Generate the word cloud
                wordcloud_future_work = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_future_work, interpolation='bilinear')
                ax.axis('off')

                return fig
            
    with ui.layout_columns():
        with ui.card():
            ui.card_header("Wordcloud for Abstracts")
            @render.plot
            def wordcloud_abstract():
                # papers = pd.read_csv("paper.csv")
                
                # Combine all keywords into a single string
                keywords_text = ' '.join(papers['Abstract Summary'].dropna())

                # Generate the word cloud
                wordcloud_abstract = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_abstract, interpolation='bilinear')
                ax.axis('off')

                return fig
            
        with ui.card():
            ui.card_header("Wordcloud for Evaluation Metrics")
            @render.plot
            def wordcloud_evaluation_metrics():
                # papers = pd.read_csv("paper.csv")
                
                # Combine all keywords into a single string
                keywords_text = ' '.join(papers['Evaluation Metrics'].dropna())

                # Generate the word cloud
                wordcloud_evaluation_metrics = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_evaluation_metrics, interpolation='bilinear')
                ax.axis('off')

                return fig
            
with ui.nav_panel("Author Network"):
    with ui.layout_columns():  
        with ui.card():
            ui.card_header("3D Network of Authors")
            @render_widget
            def plot_author_network():
                # Preprocess the author data
                author_data = papers['Authors'].dropna()
                edges = []
                for authors in author_data:
                    authors_list = [a.strip() for a in authors.split(',')]
                    for i in range(len(authors_list)):
                        for j in range(i + 1, len(authors_list)):
                            edges.append((authors_list[i], authors_list[j]))

                # Create the author graph
                author_graph = nx.Graph()
                author_graph.add_edges_from(edges)

                # Compute the positions of the nodes
                pos = nx.spring_layout(author_graph, dim=3)

                # Create a DataFrame with the positions of the nodes
                pos_df = pd.DataFrame(pos).T
                pos_df.columns = ['x', 'y', 'z']

                # Create a DataFrame with the names of the nodes
                names_df = pd.DataFrame(author_graph.nodes, columns=['Name'])

                # Merge the two DataFrames
                nodes_df = pd.concat([names_df, pos_df.reset_index(drop=True)], axis=1)

                # Create a plotly figure
                # Create a plotly figure
                fig = px.scatter_3d(nodes_df, x='x', y='y', z='z', text='Name')
                fig.update_traces(marker=dict(size=3))  # Make the dots smaller
                fig.update_layout(height=750, template='simple_white', scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=dict(eye=dict(x=0.5, y=0.5, z=0.2))  # Set default zoom level to 5x
                ))  # Turn off the axis

                # Add edges to the plot
                edge_x = []
                edge_y = []
                edge_z = []
                for edge in author_graph.edges():
                    x0, y0, z0 = pos[edge[0]]
                    x1, y1, z1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])

                fig.add_trace(
                    go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='none'
                    )
                )

                return fig