#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
import math
import geopandas as gpd
import csv
import streamlit as st
#import geoplot as gplt
from datetime import datetime
from pylab import *
from shapely.ops import nearest_points




prop_sales = pd.read_csv('clean_nyc_sale.csv')

### Streamlit 
st.title('This is My First Data App')

DATA_URL = ('clean_nyc_airbnb.csv')
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
Airbnb = load_data()

# show data on streamlit
st.write(Airbnb)



#CALCULATE ANNUAL EARNING FROM AIRBNB
#assume the given occupnacy percentage
#assume: airbnb take 15%, maintainence 15% -> profit only 70%
Airbnb['Annual_Earn'] = Airbnb['price'] * 0.75 * Airbnb['occupancy_%']/100 * 365
Airbnb = Airbnb.loc[Airbnb['sale_median'] != 0]

#DEFINE COST OR MORTGAGE RATE
#THIS WILL BE DETERMINED BASED ON SALE_PRICE
#i.e. total sale price will determine total mortgage

#Assuming fixed annual interest rate

def mortgage_calculator(p, i, n):
    
    '''
    The function calculate the periodic payment
    input: initial payment p, fixed annual interest rate i, total number of payment n
    output: perdioc payment amount
    Example:
        initial payment: p = 500,000
        fixed annual interest rate: i = 2.75
        30-year-loan with yearly payment: n = 30
    '''
    
    if i > 0: 
        A = p * i / ( 1 - ( 1 + i ) ** (-n) )
    elif i == 0:
        A = p / n
    else:
        print("print please input positive anual interest rate i")
    return A   

#assume current fixed annual interest rate for 30year loan
#assume median price per neighbourhood as initial cost
n = 30
i = 0.031
no_i = 0.0

#add a mortgage column

# create a new column and use np.select to assign values to it using our lists as arguments
Airbnb.loc[:, 'Annual_Loan_Payment'] = Airbnb['sale_median'].apply(mortgage_calculator,args=(i, n))
Airbnb.loc[:, 'Annual_Cost'] = Airbnb['sale_median'].apply(mortgage_calculator,args=(no_i, n))

conditions_main = [
    (Airbnb['Annual_Earn'] - Airbnb['Annual_Cost'] > 0),
    (Airbnb['Annual_Earn'] - Airbnb['Annual_Cost'] < 0)
]
values_main = ['Positive', 'Negative']
Airbnb.loc[:,'Performance']= np.select(conditions_main, values_main)

#calculate ROI

Airbnb.loc[:,'ROI_mortgage'] = ( Airbnb['Annual_Earn'] - Airbnb['Annual_Loan_Payment'] ) / Airbnb['Annual_Cost']
Airbnb.loc[:,'ROI'] = ( Airbnb['Annual_Earn'] - Airbnb['Annual_Cost'] ) / Airbnb['Annual_Cost']
Airbnb[['ROI_mortgage','ROI']].describe()


# We already converted last review into activity and only keep the ones that is still active
Airbnb = Airbnb.drop(columns = "last_review")

Airbnb


#divide into Types of Properties 
shared_room = Airbnb.loc[Airbnb['room_type'] == 'Shared room']
apt = Airbnb.loc[Airbnb['room_type'] == 'Entire home/apt']
private_room = Airbnb.loc[Airbnb['room_type'] == 'Private room']




Airbnb["neighbourhood_group"].unique()





# divide into Neighbourhood groups
Manhattan = Airbnb.loc[Airbnb['neighbourhood_group'] == 'Manhattan']
Brooklyn= Airbnb.loc[Airbnb['neighbourhood_group'] == 'Brooklyn']
Queens = Airbnb.loc[Airbnb['neighbourhood_group'] == 'Queens']
Staten_Island = Airbnb.loc[Airbnb['neighbourhood_group'] == 'Staten Island']
Bronx = Airbnb.loc[Airbnb['neighbourhood_group'] == 'Bronx']





# Airbnb Location Distribution 
title = 'Neighbourhood Group Location'
fig1= plt.figure(figsize=(10,6))
sns.scatterplot(Airbnb.longitude,Airbnb.latitude,hue=Airbnb.neighbourhood_group).set_title(title)
plt.ioff()
st.pyplot(fig1)



#Popularity 
title = 'Room type location per Neighbourhood Group'
fig2 = sns.catplot(x='room_type', kind="count", hue="neighbourhood_group", data=Airbnb)
plt.title(title)
plt.ioff()
st.pyplot(fig2)

# Count and turn in to list the number of Postive and Negative listing.
shared_per = shared_room['Performance'].value_counts().to_list()
private_per = private_room['Performance'].value_counts().to_list()
apt_per  = apt['Performance'].value_counts().to_list()



Manhattan_per = Manhattan['Performance'].value_counts().to_list()
Brooklyn_per = Brooklyn['Performance'].value_counts().to_list()
Queens_per = Queens['Performance'].value_counts().to_list()
Staten_Island_per = Staten_Island['Performance'].value_counts().to_list()
Bronx_per = Bronx['Performance'].value_counts().to_list()


Bronx['Performance'].value_counts()

# create a set of pie chart to compare the performance of neighbourhood group. 
fig3, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
fig3.subplots_adjust(top=0.7)
My_labels = ['Negative', "Postive" ]
My_colors = ["Pink", "lightsteelblue"]
fig3 = plt.gcf()
fig3.set_size_inches(15,4)
my_explode = (0.2, 0.2)
figure_title = fig3.suptitle('Performance based on Neighbourhood Groups', fontsize= 20)
plt.subplots_adjust(top=0.75)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# plot each pie chart in a separate subplot


ax1.pie(Manhattan_per,labels=My_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors = My_colors, explode=my_explode, )
ax1.set_title('Manhattan', pad = 15, fontsize = 15)

ax2.pie(Brooklyn_per,labels=My_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors = My_colors, explode=my_explode)
ax2.set_title('Staten_Island', pad = 15, fontsize = 15)

ax3.pie(Queens_per,labels=My_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors = My_colors, explode=my_explode)
ax3.set_title('Queens', pad = 15, fontsize = 15)

ax4.pie(Staten_Island_per,labels=My_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors = My_colors, explode=my_explode)
ax4.set_title('Brooklyn', pad = 15, fontsize = 15)

ax5.pie(Bronx_per, labels=My_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors = My_colors, explode=my_explode)
ax5.set_title('Bronx', pad = 15, fontsize = 15)

plt.show()
st.pyplot(fig3)



# create a set of pie chart to compare the performance of 3 types of properties. 
fig4, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig4.subplots_adjust(top=0.7)
shared_labels = ['Negative', "Postive" ]
private_labels = ['Negative', "Postive" ]
apt_labels = ['Positve', "Negative" ]
my_colors = ["Red", "Blue"]
apt_colors = ["Blue", "Red"]
fig4 = plt.gcf()
fig4.set_size_inches(15,4)
my_explode = (0.1, 0.1)
fig4.suptitle('Performance of properties', fontsize= 20)
plt.subplots_adjust(top=0.8)


# plot each pie chart in a separate subplot


ax1.pie(shared_per, labels=shared_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors = my_colors, explode=my_explode, )
ax1.set_title('Shared Room Performance')


ax2.pie(apt_per, labels=apt_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors = apt_colors, explode=my_explode)
ax2.title.set_text('Apartment and Entire_Room Performance')

ax3.pie(private_per, labels=private_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors, explode=my_explode)
ax3.title.set_text('Private Room Performance')

plt.show()
st.pyplot(fig4)

# In[20]:


#PLOT ANNUAL LOAN PAYMENT VS ANNUAL EARN

fig5 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig5.subplots_adjust(top=0.8)
x = np.arange(300000)
y=x

### SHARED ROOM
ax1 = fig5.add_subplot(131)
ax1 = sns.scatterplot(shared_room.Annual_Loan_Payment, shared_room.Annual_Earn, hue = shared_room.neighbourhood_group)
ax1.get_legend().remove()

plt.plot(x,y)
plt.title('Shared Room Annual Earning vs Annual Loan Payment', y = 1.05)
plt.ylabel('Annual_Earning / $')
plt.xlabel('Annual_Loan_Payment / $')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()

ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.8])


### PRIVATE ROOM
ax2 = fig5.add_subplot(132)
ax2 = sns.scatterplot(private_room.Annual_Loan_Payment, private_room.Annual_Earn, hue = private_room.neighbourhood_group)

plt.plot(x,y)

plt.title('Private Room Annual Earning vs Annual Loan Payment', y = 1.05)
plt.ylabel('Annual earning / $' )
plt.xlabel('Annual mortgage / $')
plt.grid(alpha=1,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()

ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.8])

#Edit legend
handles, labels = ax2.get_legend_handles_labels()

ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})

### ENTIRE HOME
ax3 = fig5.add_subplot(133)
ax3 = sns.scatterplot(apt.Annual_Loan_Payment, apt.Annual_Earn, hue = apt.neighbourhood_group)
ax3.get_legend().remove()

plt.plot(x,y)
plt.title('Entire Home Annual Earning vs Annual Loan Payment', y = 1.05)
plt.ylabel('Annual earning / $', x = 1.5)
plt.xlabel('Annual mortgage / $')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.8])

plt.show()
st.pyplot(fig5)

#PLOT ANNUAL COST VS ANNUAL EARN

fig6 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig6.subplots_adjust(top=0.8)

x = np.arange(100000)
y=x

### SHARED ROOM
ax1 = fig6.add_subplot(131)
ax1 = sns.scatterplot(shared_room.Annual_Cost, shared_room.Annual_Earn, hue = shared_room.neighbourhood_group)
ax1.get_legend().remove()

plt.plot(x,y)
plt.title('Shared Room Annual Earning vs Annual Cost')
plt.ylabel('Annual earning / $')
plt.xlabel('Annual cost / $')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


### PRIVATE ROOM
ax2 = fig6.add_subplot(132)
ax2 = sns.scatterplot(private_room.Annual_Cost, private_room.Annual_Earn, hue = private_room.neighbourhood_group)

plt.plot(x,y)

plt.title('Private Room Annual Earning vs Annual Cost')
plt.ylabel('Annual earning / $')
plt.xlabel('Annual cost / $')
plt.grid(alpha=.4,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# LEGEND
handles, labels = ax2.get_legend_handles_labels()

ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})


### ENTIRE HOME
ax3 = fig6.add_subplot(133)
ax3 = sns.scatterplot(apt.Annual_Cost, apt.Annual_Earn, hue = apt.neighbourhood_group)
ax3.get_legend().remove()

plt.plot(x,y)
plt.title('Entire Home Annual Earning vs Annual Cost')
plt.ylabel('Annual earning / $')
plt.xlabel('Annual cost / $')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.show()

st.pyplot(fig6)



Shared_room_pos = shared_room[shared_room["Performance"] == "Positive"]
Private_room_pos = private_room[private_room["Performance"] == "Positive"]
Apt_pos = apt[apt["Performance"] == "Positive"]







#PLOT ROI vs sale_median


fig7 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig7.subplots_adjust(top=0.8)



### SHARED ROOM
ax1 = fig7.add_subplot(131)
ax1 = sns.scatterplot(Shared_room_pos.sale_median, Shared_room_pos.ROI, hue = Shared_room_pos.neighbourhood_group)
ax1.get_legend().remove()


plt.title('Shared Room ROI vs Median Sale Price', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Median Sale Price / $')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


### PRIVATE ROOM
ax2 = fig7.add_subplot(132)
ax2 = sns.scatterplot(Private_room_pos.sale_median, Private_room_pos.ROI, hue = Private_room_pos.neighbourhood_group)



plt.title('Private Room ROI vs Median Sale Price', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Median Sale Price / $')
plt.grid(alpha=.4,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

#Edit legend
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})

### ENTIRE HOME
ax3 = fig7.add_subplot(133)
ax3 = sns.scatterplot(Apt_pos.sale_median, Apt_pos.ROI, hue = Apt_pos.neighbourhood_group)
ax3.get_legend().remove()

plt.title('Entire Home ROI vs Median Sale Price', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Median Sale Price / $')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.show()
st.pyplot(fig7)


#PLOT ROI vs price per night
fig8 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig8.subplots_adjust(top=0.8)



### SHARED ROOM
ax1 = fig8.add_subplot(131)
ax1 = sns.scatterplot(Shared_room_pos.price, Shared_room_pos.ROI, hue = Shared_room_pos.neighbourhood_group)
ax1.get_legend().remove()


plt.title('Shared Room ROI vs Airbnb Price', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Airbnb Price / $')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


### PRIVATE ROOM
ax2 = fig8.add_subplot(132)
ax2 = sns.scatterplot(Private_room_pos.price, Private_room_pos.ROI, hue = Private_room_pos.neighbourhood_group)



plt.title('Private room ROI vs Airbnb Price', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Airbnb Price / $')
plt.grid(alpha=.4,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

#Edit legend
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})

### ENTIRE HOME
ax3 = fig8.add_subplot(133)
ax3 = sns.scatterplot(Apt_pos.price, Apt_pos.ROI, hue = Apt_pos.neighbourhood_group)
ax3.get_legend().remove()

plt.title('APARTMENT AND ENTIRE HOUSE ROI vs Airbnb Price', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Airbnb Price / $')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.show()
st.pyplot(fig8)

#PLOT ROI vs Location 
fig9 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig9.subplots_adjust(top=0.8)



### SHARED ROOM
ax1 = fig9.add_subplot(131)
ax1 = sns.scatterplot(Shared_room_pos.how_far_km, Shared_room_pos.ROI, hue = Shared_room_pos.neighbourhood_group)
ax1.get_legend().remove()


plt.title('Shared Room ROI vs Their Location', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('How far from landmarks / km')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


### PRIVATE ROOM
ax2 = fig9.add_subplot(132)
ax2 = sns.scatterplot(Private_room_pos.how_far_km, Private_room_pos.ROI, hue = Private_room_pos.neighbourhood_group)



plt.title('Private room ROI vs Their Location', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('How far from landmarks / km')
plt.grid(alpha=.4,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

#Edit legend
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})

### ENTIRE HOME
ax3 = fig9.add_subplot(133)
ax3 = sns.scatterplot(Apt_pos.how_far_km, Apt_pos.ROI, hue = Apt_pos.neighbourhood_group)
ax3.get_legend().remove()

plt.title('APARTMENT AND ENTIRE HOUSE ROI vs Their Location', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('How far from landmarks / km')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.show()
st.pyplot(fig9)


#PLOT ROI vs Number of reviews
fig10 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig10.subplots_adjust(top=0.8)



### SHARED ROOM
ax1 = fig10.add_subplot(131)
ax1 = sns.scatterplot(Shared_room_pos.number_of_reviews, Shared_room_pos.ROI, hue = Shared_room_pos.neighbourhood_group)
ax1.get_legend().remove()


plt.title('Shared Room ROI vs Their Reviews', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Number of reviews / review')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


### PRIVATE ROOM
ax2 = fig10.add_subplot(132)
ax2 = sns.scatterplot(Private_room_pos.number_of_reviews, Private_room_pos.ROI, hue = Private_room_pos.neighbourhood_group)



plt.title('Private room ROI vs Their Reviews', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Number of reviews / review')
plt.grid(alpha=.4,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

#Edit legend
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})

### ENTIRE HOME
ax3 = fig10.add_subplot(133)
ax3 = sns.scatterplot(Apt_pos.number_of_reviews, Apt_pos.ROI, hue = Apt_pos.neighbourhood_group)
ax3.get_legend().remove()

plt.title('APARTMENT AND ENTIRE HOUSE ROI vs Their Reviews', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Number of reviews / review')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.show()

st.pyplot(fig10)

Airbnb.columns

#PLOT ROI vs minimum_nights
fig11 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig11.subplots_adjust(top=0.8)



### SHARED ROOM
ax1 = fig11.add_subplot(131)
ax1 = sns.scatterplot(Shared_room_pos.minimum_nights, Shared_room_pos.ROI, hue = Shared_room_pos.neighbourhood_group)
ax1.get_legend().remove()


plt.title('Shared Room ROI vs minimum nights requirement', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Mininum nights / nights')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


### PRIVATE ROOM
ax2 = fig11.add_subplot(132)
ax2 = sns.scatterplot(Private_room_pos.minimum_nights, Private_room_pos.ROI, hue = Private_room_pos.neighbourhood_group)



plt.title('Private room ROI vs minimum nights requirement', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Mininum nights / nights')
plt.grid(alpha=.4,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

#Edit legend
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})

### ENTIRE HOME
ax3 = fig11.add_subplot(133)
ax3 = sns.scatterplot(Apt_pos.minimum_nights, Apt_pos.ROI, hue = Apt_pos.neighbourhood_group)
ax3.get_legend().remove()

plt.title('APARTMENT AND ENTIRE HOUSE ROI vs minimum nights requirement', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Mininum nights / nights')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.show()
st.pyplot(fig11)

#PLOT ROI vs calculated_host_listings_count
fig12 = plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
fig12.subplots_adjust(top=0.8)



### SHARED ROOM
ax1 = fig12.add_subplot(131)
ax1 = sns.scatterplot(Shared_room_pos.calculated_host_listings_count, Shared_room_pos.ROI, hue = Shared_room_pos.neighbourhood_group)
ax1.get_legend().remove()


plt.title('Shared Room ROI vs number of their host listings', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Number of listing / properties')
plt.grid(alpha=.4,linestyle='--')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


### PRIVATE ROOM
ax2 = fig12.add_subplot(132)
ax2 = sns.scatterplot(Private_room_pos.calculated_host_listings_count, Private_room_pos.ROI, hue = Private_room_pos.neighbourhood_group)



plt.title('Private room ROI vs number of their host listings', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('Number of listing / properties')
plt.grid(alpha=.4,linestyle='--')

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

#Edit legend
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[1:], labels=labels[1:], loc=8, 
          ncol=5, bbox_to_anchor=[0.5,-.3,0,0], shadow = True, prop={"size":15})

### ENTIRE HOME
ax3 = fig12.add_subplot(133)
ax3 = sns.scatterplot(Apt_pos.calculated_host_listings_count, Apt_pos.ROI, hue = Apt_pos.neighbourhood_group)
ax3.get_legend().remove()

plt.title('APARTMENT AND ENTIRE HOUSE ROI vs number of their host listings', y=1.05)
plt.ylabel('ROI / %')
plt.xlabel('number of listings / properties')
plt.grid(alpha=.4,linestyle='--')

box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.show()
st.pyplot(fig12)






