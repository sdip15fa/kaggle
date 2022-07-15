#!/usr/bin/env python
# coding: utf-8

# # Simple ML Jump (STUDENT VERSION)
# Billy Hau - June 29, 2022
# 
# - Data Cleaning / Feature Engineering / Modeling 
# - Deployment

# In[67]:


# Warning


# In[68]:


# Import Libraries
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import multiprocessing as mp


# In[69]:


# Read CSV File
df = pd.read_csv("./data.csv")
df


# In[70]:


# What do each of the columns mean?


# In[71]:


jump = df[df['Jump'] == True]
dead = jump['Dead']
# dead.value_counts()
print(jump.shape)
for i in range(jump.shape[0]):
    # print(jump.index[i])
    try:
        if any(any(df['Dead'][jump.index[i + y] + x] for x in range(0, 11)) for y in range(0, 2)) or not any(df['Scored'][jump.index[i] + x] for x in range(0, 5)):
            df.drop(df[df.index == jump.index[i]].index, inplace=True)
    except:
        pass
df


# In[72]:


# dead.value_counts()
print(df.shape)
for i in range(df.shape[0]):
    # print(jump.index[i])
    try:
        if any(df['Dead'][df.index[i] + x] for x in range(0, 6)):
            df.drop(df[df.index == df.index[i]].index, inplace=True)
    except:
        pass
df


# In[73]:


# Are there data we don't care about?
df.drop(df[df['Dead'] == True].index, inplace=True)
df = df.drop(["Dead", "Scored"], inplace=False, axis=1)
df


# In[74]:


jumpCount = df[df['Jump'] == True].shape[0]
jumpCount


# In[75]:


noJumpCount = df[df['Jump'] == False].shape[0]
noJumpCount


# In[76]:


df_toDrop = df[df['Jump'] == False].sample(noJumpCount - jumpCount)
df.drop(df_toDrop.index, inplace=True)


# In[77]:


# Create the Input DataFrame
X = df[['Bar 1 Distance', 'Bar 1 Speed', 'Bar 2 Distance', 'Bar 2 Speed']]
X = X.to_numpy()


# In[78]:


# Create the Output DataFrame and Map True / False to 1 / 0
Y = df[['Jump']]


def bool2Num(row):
    if row['Jump']:
        return 1
    else:
        return 0


Y['Jump'] = Y.apply(bool2Num, axis='columns')
Y = Y['Jump'].to_numpy()


# In[79]:


# Train Test Split with test size set to 20%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[80]:


model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[81]:


predict = model.predict(x_test)


# In[82]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[83]:


# Print Out Prediction Accuraccy
accuracy_score(y_test, predict)


# In[84]:


# Try a Different Model (Random Forest)
"""model2 = ________________________________
model2.__________________________

predict = ___________________________
accuracy_score(_________________, ________________)"""


# # Deployment

# In[85]:


# Command to Open SIMPLE ML Jump 2 (for Example)

# Windows OS
#env_path = "D:\\User\\Desktop\\10botics Data Science\\ML Game\\SimpleMLJump2 Builds\\Windows\\Simple ML Jump 2.exe"

# MacOS
#env_path = 'open -n "/Users/billwaa/Desktop/ML Game/Simple ML Jump 2.app"' 

env_path = "../Linux/SimpleMLJump2.x86_64"


# In[86]:


# Import Libraries
import socket
import struct
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import time
import subprocess
import platform
import collect
import multiprocessing as mp


# Find Open Socket Ports
def findOpenSockets(size: int):
    sock = []
    port = []

    for i in range(size):
        sock.append(socket.socket())
        sock[i].bind(('localhost', 0))
        port.append(sock[i].getsockname()[1])

    for i in range(len(sock)):
        sock[i].close()

    return port


# Find Open Ports and Launch Game Environment
envNum = 1
openPorts = findOpenSockets(envNum * 2)
gamePort = openPorts[0]
apiPort = openPorts[1]

# Determine OS and Launch 
if platform.system() == 'Darwin':
    env_path += f" --args --apiPort {str(apiPort)} --gamePort {str(gamePort)} --small false"
    process = subprocess.Popen([env_path], shell=True)
else:
    process = subprocess.Popen([env_path, '--apiPort', str(apiPort), '--gamePort', str(gamePort), '--small', 'false'])

# Establish UDP Network Client
localIP = "localhost"
bufferSize = 1024
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDPServerSocket.bind((localIP, apiPort))

# Set UDP Timeout
UDPServerSocket.settimeout(3)

# Wait for Game Environment to Open
t0 = time.time()

while (time.time() - t0 < 5):
    pass


# Extract Data
def extractData(data):
    dist1 = struct.unpack('f', data[:4])
    speed1 = struct.unpack('f', data[4:8])
    dist2 = struct.unpack('f', data[8:12])
    speed2 = struct.unpack('f', data[12:16])

    onGround = True if (data[16] & (1 << 0)) == 1 else False
    toJump = True if (data[17] & (1 << 0)) == 1 else False
    isDead = True if (data[18] & (1 << 0)) == 1 else False
    scored = True if (data[19] & (1 << 0)) == 1 else False

    dat = [dist1[0], speed1[0], dist2[0], speed2[0], toJump, onGround, isDead,
           scored]  # Swap Order to Accomadate Old Model
    df_temp = pd.DataFrame(dat).transpose()

    return df_temp


# Reset Simulation
UDPServerSocket.sendto(bytes.fromhex('07 00'), (localIP, gamePort))


# In[ ]:


from os.path import exists

df2 = pd.DataFrame()
prevJump = False
# Process Loop - Retrieve Data from Simulation, Run Through Model, Output Command
while True:
    try:
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        t0 = time.time()
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        df_temp = extractData(message)
        X = df_temp.iloc[:, :4]
        X = X.astype(float)
    # Data PreProcessing Function Go Here
    # Model Prediction Function Go Here
        predict = model.predict(X)
        print(predict, end='\r')
    # If Jump Decision is Made, Send to Game
        if predict[0] > 0.5 and not (not df_temp.iloc[0, 5] and (df_temp.iloc[0, 2] - df_temp.iloc[0, 0]) >= 15):
            # Jump Command
            if not prevJump:
                UDPServerSocket.sendto(bytes.fromhex(
                    '06 01'), (localIP, gamePort))
                prevJump = True
            else:
                prevJump = False
        else:
            prevJump = False
        if df_temp.iloc[0, 6]:
            # df2.columns = ['Bar 1 Distance', 'Bar 1 Speed', 'Bar 2 Distance',
            #               'Bar 2 Speed', 'Jump', 'Grounded', 'Dead', 'Scored']
            """try:
                if exists("data.csv"):
                    pd.concat([pd.read_csv("data.csv"),
                               df2]).to_csv("data.csv", index=False)
                else:
                    df2.to_csv("data.csv", index=False)

                score = df2['Scored'].value_counts()[True]

                def clean(x): return x > 0

                dfScore = pd.DataFrame(
                    [[score, (filter(clean, df2['Bar 2 Distance']) - filter(clean, df2['Bar 1 Distance'])).mean(), (filter(clean, df2['Bar 1 Speed']).mean() + filter(clean, df2['Bar 2 Speed']).mean()) / 2]], columns=['Score', 'Bar Distance', 'Bar Speed'])

                if exists("score.csv"):
                    pd.concat([pd.read_csv("score.csv"), dfScore]
                              ).to_csv("score.csv", index=False)
                else:
                    dfScore.to_csv("score.csv", index=False)
            except Exception as e:
                print(e)
                pass"""
            df2 = pd.DataFrame()
            UDPServerSocket.sendto(bytes.fromhex('07 00'), (localIP, gamePort))

        # df2 = pd.concat([df2, df_temp])
    except Exception as e:
        print(e)
        break


# In[ ]:


# Export Data to CSV
df2.columns = ['Bar 1 Distance', 'Bar 1 Speed', 'Bar 2 Distance',
               'Bar 2 Speed', 'Jump', 'Grounded', 'Dead', 'Scored']
try:
    pd.concat([pd.read_csv("data.csv"),
               df2]).to_csv("data.csv", index=False)
except:
    df2.to_csv("data.csv", index=False)


# In[ ]:


# Close UDP Port When Game is Closed
UDPServerSocket.close()

