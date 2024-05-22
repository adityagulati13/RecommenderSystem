import streamlit as st
import pandas as pd
import pickle
from recommender import pop_rem
def recommend(course):
    # getting the index number of this course
    course_index = courses[courses['course_title'] == course].index[0]
    # now finding index of similar courses
    distances = similarity[course_index]
    #     gettting a tuplle of similar courses
    courses_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:5]
    # making a list of recommended courses
    recommended_courses=[]

    for i in courses_list:
        recommended_courses.append(courses.iloc[i[0]]['course_title'])
    return recommended_courses

similarity=pickle.load(open('similaritynew1.pkl','rb'))
courses_dict=pickle.load(open('courses_dictnew1.pkl','rb'))
courses=pd.DataFrame(courses_dict)
print(courses.shape)
st.title('UDEMY COURSE RECOMMENDER')
selected_course_name=st.selectbox("SELECT THE COURSE NAME",courses['course_title'].values)
 #adding recommend button
if st.button("RECOMMEND"):
    recommendations=recommend(selected_course_name)
    for i in recommendations:
        st.write(i)


if st.button("OTHER TRENDING COURSES ON UDEMY"):
    trending=pop_rem(courses,7)
    st.write(trending)



st.text("DEVELOPED BY- Aditya Gulati")