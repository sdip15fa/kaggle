import pandas as pd
import time
import struct

def collect(UDPServerSocket):    
    def collectFunc():
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
    
        # %%
    
        # Collect Data While Game is in Progress
    
        df = pd.DataFrame()
        bufferSize  = 1024
    
        while True:
            try:
                bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
                t0 = time.time()
                message = bytesAddressPair[0]
                address = bytesAddressPair[1]
                df_temp = extractData(message)
                df = pd.concat([df, df_temp])
    
                print(df_temp, end='\r')
            except Exception as e:
                print(e)
                break
        # %%
    
        # Set DataFrame Column Header
        df.columns = ['Bar 1 Distance', 'Bar 1 Speed', 'Bar 2 Distance', 'Bar 2 Speed', 'Jump', 'Grounded', 'Dead',
                      'Scored']
        # %%
    
        pd.read_csv("data.csv").append(df).to_csv("./data.csv", index=False)
    return collectFunc
