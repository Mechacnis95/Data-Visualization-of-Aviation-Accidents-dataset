
# coding: utf-8

# # STEP BY STEP BEGINEERS GIUDE FOR DATA ANALYSIS OF AVIATION ACCIDENTS DATA by SWAPNIL GHARAT 

# This code contains some basic intructions for extreme begineers
# Note:Extreme begineers please read the instructions carefully I have added some baic examples of the functions
# 
# Note: Pros please neglect the isntructions and suggest any improvements

# In[1]:


##Importing the packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib import cm
from sklearn.cluster import KMeans


# Importing data
# Select your own path from your database.
# 
# 
# Note: Some of you might get this Error ('utf-8' codec can't decode byte 0xd3 in position 8: invalid continuation byte)
# 
# Solution: add, engine='python' after the path
# 

# In[2]:


train=pd.read_csv('/Users/swapnilgharat/Desktop/KAGGLE PROJECTS/CONSOLE GAMES/Aviation Dataset/AviationData.csv', engine='python')


# Take a sneak peak at the data, understanding the data.

# In[3]:


train.head()


# In[4]:


train.tail()


# Check data types of each columns

# In[5]:


train.dtypes
##you can observe that Event.Date is not in the right format,hence we need to convert it to date/time format
#Check the next step


# In[6]:


train['Event.Date']=pd.to_datetime(train['Event.Date'],format='%Y.%m.%d')


# In[7]:


##for newbies, if you want to check the month ,year (try using these commands)
#train['Event.Date'].dt.month
#train['Event.Date'].dt.year


# In[8]:


##Converting the numerical columns( to get adequate numeric values ,Errors:'coerce' will convert illegal values as NAN and others as standard numeric) )
train['Latitude'] = pd.to_numeric(train.Latitude, errors='coerce')
train['Longitude'] = pd.to_numeric(train.Longitude, errors='coerce')
train = train.dropna(axis=0, subset=['Latitude', 'Longitude'])


# In[9]:


train.dtypes


# In[10]:


##Printing out columns
for i in train.columns:
    print(i)


# We need to add a month column for further analysis;
# Henceforth use map function on the 'Event.Date' Column
# What is a map function?
# map() function returns a list of the results after applying the given function to each item of a given iterable (list, tuple etc.) Syntax : map(fun, iter) Parameters : fun : It is a function to which map passes each element of given iterable.
# 
# 

# In[11]:


#Example( Try running this below example for basic understanding of the map function:
#r=[1,2,34,5,5,6,6]
#result=list(map(lambda x: x**2,r))
#result


# In[12]:


train['month'] = train['Event.Date'].map(lambda x: x.month)


# In[13]:


#after this we will get month for each specific dates from Event.date
train['month']


# In[14]:


train.head()


# In[15]:


##plotting the bar chart
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(8, 8))
fig.subplots_adjust(hspace=0.8)
total_month = train['month'].value_counts()
print(total_month)
train['month'].value_counts().sort_index().plot(ax=axes[0], kind='bar', title='month-wise accidents')


# In[16]:


train.Make


# AS you can observe after seeing these values the characters are not uniform with uppercase and lowercase letters.
# Hence,we need to get them into lower cases/upper casses and get rid of white spaces between them.

# In[17]:


train['cleaned.make'] = train['Make'].map(lambda x: "{}".format(x).lower().strip())
#strip() fucntion will remove the white spaces.


# # Data Visualization

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15, 15))
fig.subplots_adjust(hspace=0.6)
colors = ['#99cc33', '#a333cc', '#333dcc']
train['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,0], kind='bar', title='Phase of Flight')
train['Weather.Condition'].value_counts().plot(ax=axes[1,0], kind='pie', colors=colors, title='Weather Condition')
train['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,1], kind='pie', title='Phase of Flight')
##go through the code step by step to understand the concept of presenatation


# # You will need to import Basemap on your system to use the further code

# In[ ]:



from bokeh.io import show
from bokeh.sampledata import us_states
from bokeh.plotting import figure
##intalling basemap
get_ipython().system('conda install basemap')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


# In[ ]:


y


# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
north, south, east, west = 71.39, 24.52, -66.95, 172.5
#m = Basemap(
#    projection='lcc',
#    llcrnrlat=south,
#    urcrnrlat=north,
#    llcrnrlon=west,
#    urcrnrlon=east,
#    lat_1=33,
#    lat_2=45,
#    lon_0=-95,
#    resolution='l')
m = Basemap(llcrnrlon=-145.5,llcrnrlat=1.0,urcrnrlon=-2.566,urcrnrlat=46.352,
            rsphere=(6378137.00,6356752.3142),
            resolution='l',area_thresh=1000.0,projection='lcc',
            lat_1=50.0,lon_0=-107.0,ax=ax)
x, y = m(train['Longitude'].values, train['Latitude'].values)
m.drawcoastlines()
m.drawcountries()
m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd)


# # Using Kmeans-Clustering

# In[ ]:


latlon = train[['Longitude', 'Latitude']]
latlon.head()


# In[ ]:


kmeans = KMeans(n_clusters=50)
kmodel = kmeans.fit(latlon)
centroids = kmodel.cluster_centers_


# In[ ]:


centroids
lons, lats = zip(*centroids)
print(lats)
print(lons)


# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
north, south, east, west = 71.39, 24.52, -66.95, 172.5
m = Basemap(llcrnrlon=-145.5,llcrnrlat=1.0,urcrnrlon=-2.566,urcrnrlat=46.352,
            rsphere=(6378137.00,6356752.3142),
            resolution='l',area_thresh=1000.0,projection='lcc',
            lat_1=50.0,lon_0=-107.0,ax=ax)
x, y = m(train['Longitude'].values, train['Latitude'].values)
m.drawcoastlines()
m.drawcountries()
m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd)
cx, cy = m(lons, lats)
m.scatter(cx, cy, 3, color='g')


# Let's riff on this map a bit more (still going to be population-centric, but there's some interesting stuff happening in Alaska)

# In[ ]:


from bokeh.sampledata import us_states
us_states = us_states.data.copy()
state_xs = [us_states[code]["lons"] for code in us_states]
state_ys = [us_states[code]["lats"] for code in us_states]
p = figure(title="Aviation Incidents and Centroids", 
           toolbar_location="left", plot_width=1100, plot_height=700)
p.patches(state_xs, state_ys, fill_alpha=0.0,
    line_color="#884444", line_width=1.5)
p.circle(train['Longitude'].data, train['Latitude'].data, size=8, color='navy', alpha=1)
p.circle(lons, lats, size=8, color='navy', alpha=1)
show(p)

