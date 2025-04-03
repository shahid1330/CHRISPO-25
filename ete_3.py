import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from wordcloud import WordCloud
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import base64
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="CHRISPO '25 Inter College Tournament",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display header
st.title("CHRISPO '25 Inter College Tournament")

# Sidebar navigation
st.sidebar.image("https://img.freepik.com/free-vector/gradient-national-sports-day-illustration_23-2148995776.jpg", use_column_width=True)
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", 
                                ["Home", "Dataset Generation", "Dashboard", "Text Analysis", "Image Processing"])

# Define constants for dataset generation
SPORTS = ["Cricket", "Football", "Basketball", "Volleyball", "Tennis", 
          "Badminton", "Table Tennis", "Athletics", "Swimming", "Chess"]
COLLEGES = ["St. Xavier's College", "Lady Shri Ram College", "Loyola College", 
            "Christ University", "Hansraj College", "Miranda House", 
            "Presidency College", "Mount Carmel College", "Fergusson College", 
            "St. Stephen's College", "SRCC", "Hindu College", "IIT Delhi", 
            "NIT Trichy", "BITS Pilani"]
STATES = ["Delhi", "Maharashtra", "Tamil Nadu", "Karnataka", "Telangana", 
          "West Bengal", "Uttar Pradesh", "Gujarat", "Punjab", "Rajasthan"]
FEEDBACK_TEMPLATES = [
    "The {sport} event was {adj}. I {verb} the {noun}.",
    "I found the {sport} competition to be {adj}. The {noun} was {adj2}.",
    "{adj} {sport} tournament! The {noun} could be {adj2} though.",
    "The {sport} event had {adj} organization but {adj2} {noun}.",
    "Really {adj} experience at the {sport} event. Would {verb} again next year."
]
ADJECTIVES = ["amazing", "fantastic", "good", "excellent", "outstanding", "decent", 
              "average", "poor", "challenging", "exciting", "boring", "innovative", 
              "well-organized", "chaotic", "memorable", "disappointing", "inspiring"]
NOUNS = ["organization", "facilities", "tournament structure", "schedule", "venue", 
         "competition level", "equipment", "management", "referees", "food services"]
VERBS = ["enjoyed", "appreciated", "loved", "disliked", "hated", "noticed", "admired", 
         "criticized", "recommend", "participate in"]

def generate_dataset(num_participants=300):
    """Generate a random dataset for CHRISPO '25 tournament participants"""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate start date for the tournament
    start_date = datetime(2025, 3, 15)
    
    # Initialize lists to store data
    data = []
    
    # Generate data for each participant
    for i in range(num_participants):
        participant_id = f"CHRP{i+1:03d}"
        name = f"Participant {i+1}"
        age = random.randint(18, 25)
        gender = random.choice(["Male", "Female"])
        college = random.choice(COLLEGES)
        state = random.choice(STATES)
        sport = random.choice(SPORTS)
        day_num = random.randint(1, 5)
        date = (start_date + timedelta(days=day_num-1)).strftime("%Y-%m-%d")
        
        # Generate feedback
        template = random.choice(FEEDBACK_TEMPLATES)
        adj = random.choice(ADJECTIVES)
        adj2 = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        verb = random.choice(VERBS)
        
        feedback = template.format(sport=sport, adj=adj, adj2=adj2, noun=noun, verb=verb)
        
        # Calculate performance score
        performance_score = round(random.uniform(5.0, 10.0), 1)
        
        # Append participant data
        data.append({
            "Participant_ID": participant_id,
            "Name": name,
            "Age": age,
            "Gender": gender,
            "College": college,
            "State": state,
            "Sport": sport,
            "Day": day_num,
            "Date": date,
            "Performance_Score": performance_score,
            "Feedback": feedback
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

# Function to create wordcloud
def generate_wordcloud(text):
    stop_words = set(stopwords.words('english'))
    
    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         stopwords=stop_words,
                         min_font_size=10).generate(text)
    
    # Display the generated image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function for image processing
def process_image(image, filter_type):
    if filter_type == "Grayscale":
        return ImageOps.grayscale(image)
    elif filter_type == "Blur":
        return image.filter(ImageFilter.BLUR)
    elif filter_type == "Contour":
        return image.filter(ImageFilter.CONTOUR)
    elif filter_type == "Enhance Contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)
    elif filter_type == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == "Edge Enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    else:
        return image  # Original image

# Function to load example sports images
def get_sample_images():
    # Dictionary mapping days to lists of sample sports images
    sample_images = {
        1: [
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_1.jpg",
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_2.jpg"  
        ],
        2: [
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_3.jpg",  
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_4.jpg"   
        ],
        3: [
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_5.jpg", 
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_6.jpg"   
        ],
        4: [
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_7.jpg",  
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_8.jpg"   
        ],
        5: [
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_9.jpg",  
            "E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/4105749.jpg"  
        ]
    }
    return sample_images

# Home page content
if selected_page == "Home":
    st.header("Welcome to CHRISPO '25 Data Analysis Platform")
    
    st.write("""
    This application provides comprehensive analysis and visualization tools for the CHRISPO '25 Inter-College Tournament.
    
    ### Features:
    - **Dataset Generation**: Create a dataset with 300 participants across 10 sports events over 5 days
    - **Dashboard**: Visualize participation trends through interactive charts and filters
    - **Text Analysis**: Process participant feedback and generate sport-wise word clouds
    - **Image Processing**: View and apply custom filters to sports event photos
    
    Use the sidebar to navigate through different sections of the application.
    """)
    
    st.image("E:/Christ University/Trimester 3/Advanced Python/Code/ETE_3/sports_event.jpg", use_column_width=True)

# Dataset Generation page content
elif selected_page == "Dataset Generation":
    st.header("Dataset Generation")
    
    st.write("""
    Generate a dataset for CHRISPO '25 participants. The dataset includes information about 300 participants 
    across 10 different sports over 5 days, including feedback from each participant.
    """)
    
    if st.button("Generate Dataset"):
        # Generate dataset
        df = generate_dataset()
        
        # Store dataset in session state
        st.session_state['dataset'] = df
        
        # Display sample of the dataset
        st.success(f"Dataset generated successfully with {len(df)} entries!")
        st.subheader("Sample Data (First 10 entries)")
        st.write(df.head(10))
        
        # Download link
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="chrispo_25_dataset.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # If dataset already exists in session state, show it
    if 'dataset' in st.session_state:
        st.subheader("Current Dataset (First 10 entries)")
        st.write(st.session_state['dataset'].head(10))
        
        # Statistics
        st.subheader("Dataset Statistics")
        df = st.session_state['dataset']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Participants:** {len(df)}")
            st.write(f"**Unique Sports:** {df['Sport'].nunique()}")
            st.write(f"**Unique Colleges:** {df['College'].nunique()}")
        
        with col2:
            st.write(f"**Number of Days:** {df['Day'].nunique()}")
            st.write(f"**Gender Distribution:** M: {len(df[df['Gender']=='Male'])}, F: {len(df[df['Gender']=='Female'])}")
            st.write(f"**Average Age:** {df['Age'].mean():.2f} years")

# Dashboard page content
elif selected_page == "Dashboard":
    st.header("Interactive Dashboard")
    
    if 'dataset' not in st.session_state:
        st.warning("Please generate the dataset first from the Dataset Generation page.")
        if st.button("Generate Dataset Now"):
            st.session_state['dataset'] = generate_dataset()
            st.success("Dataset generated successfully!")
    else:
        df = st.session_state['dataset']
        
        # Filters
        st.subheader("Filter Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_sports = st.multiselect("Select Sports", options=sorted(df['Sport'].unique()), default=[])
        
        with col2:
            selected_states = st.multiselect("Select States", options=sorted(df['State'].unique()), default=[])
        
        with col3:
            selected_colleges = st.multiselect("Select Colleges", options=sorted(df['College'].unique()), default=[])
        
        # Apply filters
        filtered_df = df.copy()
        if selected_sports:
            filtered_df = filtered_df[filtered_df['Sport'].isin(selected_sports)]
        if selected_states:
            filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
        if selected_colleges:
            filtered_df = filtered_df[filtered_df['College'].isin(selected_colleges)]
        
        # Check if data is available after filtering
        if filtered_df.empty:
            st.warning("No data available with the selected filters.")
        else:
            st.success(f"Showing data for {len(filtered_df)} participants")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sports-wise", "Day-wise", "College-wise", "State-wise", "Performance"])
            
            # Tab 1: Sports-wise Participation
            with tab1:
                st.subheader("Sports-wise Participation")
                
                # Chart 1: Sports Distribution
                sports_count = filtered_df['Sport'].value_counts().reset_index()
                sports_count.columns = ['Sport', 'Count']
                
                fig = px.bar(sports_count, x='Sport', y='Count', 
                            title='Number of Participants by Sport',
                            color='Sport', text='Count')
                fig.update_layout(xaxis_title='Sport', yaxis_title='Number of Participants')
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart 2: Gender distribution by sport
                gender_sport = pd.crosstab(filtered_df['Sport'], filtered_df['Gender'])
                
                fig = px.bar(gender_sport, title='Gender Distribution by Sport',
                            barmode='group', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
                fig.update_layout(xaxis_title='Sport', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Day-wise Participation
            with tab2:
                st.subheader("Day-wise Participation")
                
                # Chart 3: Day-wise participation
                day_count = filtered_df['Day'].value_counts().sort_index().reset_index()
                day_count.columns = ['Day', 'Count']
                
                fig = px.line(day_count, x='Day', y='Count', 
                             title='Participation Trend Over Days',
                             markers=True)
                fig.update_layout(xaxis_title='Day', yaxis_title='Number of Participants')
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart 4: Sport distribution by day
                sport_day = pd.crosstab(filtered_df['Day'], filtered_df['Sport'])
                
                fig = px.area(sport_day, title='Sport Distribution by Day')
                fig.update_layout(xaxis_title='Day', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 3: College-wise Participation
            with tab3:
                st.subheader("College-wise Participation")
                
                # Chart 5: Top colleges by participation
                college_count = filtered_df['College'].value_counts().nlargest(10).reset_index()
                college_count.columns = ['College', 'Count']
                
                fig = px.bar(college_count, x='Count', y='College', 
                            title='Top 10 Colleges by Participation',
                            orientation='h', color='Count', text='Count')
                fig.update_layout(yaxis_title='', xaxis_title='Number of Participants')
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart 6: Sport preference by top colleges
                top_colleges = college_count['College'].tolist()
                college_sport_df = filtered_df[filtered_df['College'].isin(top_colleges)]
                college_sport = pd.crosstab(college_sport_df['College'], college_sport_df['Sport'])
                
                fig = px.imshow(college_sport, title='Sport Preference by Top Colleges',
                               color_continuous_scale='viridis')
                fig.update_layout(xaxis_title='Sport', yaxis_title='College')
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: State-wise Participation
            with tab4:
                st.subheader("State-wise Participation")
                
                # Chart 7: State-wise participation
                state_count = filtered_df['State'].value_counts().reset_index()
                state_count.columns = ['State', 'Count']
                
                fig = px.pie(state_count, values='Count', names='State', 
                            title='Participation by State',
                            hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart 8: Sport preference by state
                state_sport = pd.crosstab(filtered_df['State'], filtered_df['Sport'])
                
                fig = px.bar(state_sport, title='Sport Preference by State',
                            barmode='stack')
                fig.update_layout(xaxis_title='State', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 5: Performance Analysis
            with tab5:
                st.subheader("Performance Analysis")
                
                # Chart 9: Average performance by sport
                perf_sport = filtered_df.groupby('Sport')['Performance_Score'].mean().reset_index()
                perf_sport = perf_sport.sort_values('Performance_Score', ascending=False)
                
                fig = px.bar(perf_sport, x='Sport', y='Performance_Score',
                            title='Average Performance Score by Sport',
                            color='Performance_Score', text=perf_sport['Performance_Score'].round(2))
                fig.update_layout(xaxis_title='Sport', yaxis_title='Average Score')
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart 10: Performance distribution
                fig = px.histogram(filtered_df, x='Performance_Score',
                                  title='Distribution of Performance Scores',
                                  nbins=20, color_discrete_sequence=['green'])
                fig.update_layout(xaxis_title='Performance Score', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)

# Text Analysis page content
elif selected_page == "Text Analysis":
    st.header("Feedback Text Analysis")
    
    if 'dataset' not in st.session_state:
        st.warning("Please generate the dataset first from the Dataset Generation page.")
        if st.button("Generate Dataset Now"):
            st.session_state['dataset'] = generate_dataset()
            st.success("Dataset generated successfully!")
    else:
        df = st.session_state['dataset']
        
        # Select sport for analysis
        selected_sport = st.selectbox("Select Sport for Feedback Analysis", 
                                     options=["All Sports"] + sorted(df['Sport'].unique().tolist()))
        
        if selected_sport == "All Sports":
            feedback_data = df
            title = "All Sports Feedback"
        else:
            feedback_data = df[df['Sport'] == selected_sport]
            title = f"{selected_sport} Feedback"
        
        # Generate wordcloud
        if not feedback_data.empty:
            combined_feedback = " ".join(feedback_data['Feedback'].tolist())
            st.subheader(f"Word Cloud for {title}")
            wordcloud_fig = generate_wordcloud(combined_feedback)
            st.pyplot(wordcloud_fig)
            
            # Show common words
            st.subheader(f"Common Words in {title}")
            
            # Process text to find common words
            def process_text(text):
                stop_words = set(stopwords.words('english'))
                words = text.lower().split()
                words = [word.strip('.,!?"\'()[]{}:;') for word in words]
                words = [word for word in words if word and word not in stop_words and len(word) > 2]
                return words
            
            all_words = []
            for feedback in feedback_data['Feedback']:
                all_words.extend(process_text(feedback))
            
            word_counts = Counter(all_words)
            common_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
            
            fig = px.bar(common_words, x='Word', y='Count',
                        title=f'Top 20 Common Words in {title}',
                        color='Count')
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare feedback between sports
            if selected_sport != "All Sports":
                st.subheader(f"Feedback Comparison: {selected_sport} vs. Other Sports")
                
                # Average sentiment score (using length as a simple proxy)
                sport_feedback = df[df['Sport'] == selected_sport]['Feedback']
                other_feedback = df[df['Sport'] != selected_sport]['Feedback']
                
                sport_lengths = [len(text) for text in sport_feedback]
                other_lengths = [len(text) for text in other_feedback]
                
                comparison_data = pd.DataFrame({
                    'Category': [selected_sport, 'Other Sports'],
                    'Avg. Feedback Length': [sum(sport_lengths)/len(sport_lengths), 
                                            sum(other_lengths)/len(other_lengths)]
                })
                
                fig = px.bar(comparison_data, x='Category', y='Avg. Feedback Length',
                            title='Average Feedback Length Comparison',
                            color='Category')
                st.plotly_chart(fig, use_container_width=True)
                
                # Word frequency comparison
                sport_words = []
                for feedback in sport_feedback:
                    sport_words.extend(process_text(feedback))
                
                other_words = []
                for feedback in other_feedback:
                    other_words.extend(process_text(feedback))
                
                # Get word frequencies
                sport_freq = Counter(sport_words)
                other_freq = Counter(other_words)
                
                # Normalize by total words
                sport_total = sum(sport_freq.values())
                other_total = sum(other_freq.values())
                
                common_words_all = set(list(sport_freq.keys())[:20] + list(other_freq.keys())[:20])
                
                comparison_words = []
                for word in common_words_all:
                    sport_pct = sport_freq.get(word, 0) / sport_total * 100
                    other_pct = other_freq.get(word, 0) / other_total * 100
                    comparison_words.append({
                        'Word': word,
                        selected_sport: sport_pct,
                        'Other Sports': other_pct
                    })
                
                word_comp_df = pd.DataFrame(comparison_words)
                
                # Select top words for visualization
                top_words = word_comp_df.nlargest(10, [selected_sport, 'Other Sports'])
                
                # Melt dataframe for plotting
                melted_df = pd.melt(top_words, id_vars=['Word'], 
                                   value_vars=[selected_sport, 'Other Sports'],
                                   var_name='Category', value_name='Percentage')
                
                fig = px.bar(melted_df, x='Word', y='Percentage', color='Category',
                            title='Word Frequency Comparison (% of total words)',
                            barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No feedback data available for the selected sport.")

# Image Processing page content
elif selected_page == "Image Processing":
    st.header("Image Processing Module")
    
    # Get sample images
    sample_images = get_sample_images()
    
    # Select day for gallery
    selected_day = st.slider("Select Day for Image Gallery", min_value=1, max_value=5, value=1)
    
    # Display gallery for selected day
    st.subheader(f"Day {selected_day} Image Gallery")
    
    # Create columns for images
    cols = st.columns(2)
    
    for i, img_url in enumerate(sample_images[selected_day]):
        with cols[i % 2]:
            st.image(img_url, caption=f"Sport Image {i+1}", use_column_width=True)
    
    # Custom image processing section
    st.subheader("Custom Image Processing")
    
    # Option to upload an image
    uploaded_file = st.file_uploader("Upload your own sport image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Select filter to apply
        filter_type = st.selectbox("Select Filter", 
                                  ["Original", "Grayscale", "Blur", "Contour", 
                                   "Enhance Contrast", "Sharpen", "Edge Enhance", "Emboss"])
        
        # Apply filter
        processed_image = process_image(image, filter_type)
        
        # Display processed image
        st.image(processed_image, caption=f"Processed Image ({filter_type})", use_column_width=True)
        
        # Download processed image
        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        btn = st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name=f"processed_image_{filter_type}.png",
            mime="image/png"
        )
    else:
        # Use sample image for demonstration
        st.write("Or try processing our sample image:")
        sample_img_url = sample_images[selected_day][0]
        
        # Download and process sample image
        import requests
        from PIL import Image
        from io import BytesIO
        
        try:
            response = requests.get(sample_img_url)
            sample_image = Image.open(BytesIO(response.content))
            
            # Select filter to apply
            filter_type = st.selectbox("Select Filter", 
                                      ["Original", "Grayscale", "Blur", "Contour", 
                                       "Enhance Contrast", "Sharpen", "Edge Enhance", "Emboss"])
            
            # Apply filter
            processed_sample = process_image(sample_image, filter_type)
            
            # Display processed image
            st.image(processed_sample, caption=f"Processed Sample Image ({filter_type})", use_column_width=True)
        except:
            st.error("Error loading sample image. Please try uploading your own image.")