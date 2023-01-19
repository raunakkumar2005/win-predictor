import streamlit as st
import pickle
import pandas as pd
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Giants',
    'Lucknow Supergiants'
]
cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
 'Delhi' ,'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai' ,'Cape Town'
 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg',
 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
 'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah',  'Mohali',
 'Bengaluru']
pipe = pickle.load(open('pip.pkl','rb'))


st.title('IPL win predictor')
col1,col2= st.columns(2)

with col1:
    batting_team = st.selectbox('select the bowling team',sorted(teams))
with col2:
    bowling_team = st.selectbox('select the batting team',sorted(teams))

selected_city = st.selectbox('Select the venue',sorted(cities))
target = st.number_input('target')
col3,col4,col5 = st.columns(3)


with col3:
    overs = st.number_input('Overs completed')
with col4:
    score = st.number_input('Score')
with col5:
    wickets = st.number_input('wickets')


if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets_left],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team +'-'+ str(round(loss*100)) + '%')
    st.header(bowling_team +'-'+ str(round(win*100)) + '%')


