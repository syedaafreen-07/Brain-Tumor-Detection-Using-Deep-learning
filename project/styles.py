
import streamlit as st

def load_css():

    st.markdown("""
    <style>

    /* MAIN BACKGROUND */
   .stApp{
    background: linear-gradient(to right, #D1E8F4, #DCEEFF);
    color: #1E3A5F;
}

    
    
    section[data-testid="stSidebar"]{
        background: #19405c;
        color: #c5d6ed;
    }

    section[data-testid="stSidebar"] *{
        color: #c5d6ed !important;
    }

   
    /* HEADINGS */
    h1,h2,h3,h4{
        color: #c5d6ed;
        font-family: 'Segoe UI';
    }

    /* METRIC CONTAINERS */
    [data-testid="metric-container"]{
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(111,176,226,0.2);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* DATAFRAME */
    [data-testid="stDataFrame"]{
        background-color:  #CFE8FA;
        border-radius: 15px;
        overflow: hidden;
    }

    /* BUTTON */
    .stButton>button{
        background: linear-gradient(to right,#6FB0E2,#A8D5F2);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }

    /* UPLOAD BOX */
    [data-testid="stFileUploader"]{
        background-color: rgba(255,255,255,0.5);
        border-radius: 15px;
        padding: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

def style_graph(ax, fig, title):

    ax.set_facecolor("#F4F9FF")

    fig.patch.set_facecolor("#DCEEFF")

    ax.tick_params(colors='#1E3A5F')

    ax.xaxis.label.set_color('#1E3A5F')
    ax.yaxis.label.set_color('#1E3A5F')

    ax.title.set_color('#1E3A5F')

    ax.grid(True, linestyle='--', alpha=0.2)

    ax.set_title(title, fontsize=15, weight='bold')

    for spine in ax.spines.values():
        spine.set_color("#B6D7F2")