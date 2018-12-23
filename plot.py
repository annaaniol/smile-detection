import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv('smile_records.csv')

smile_ratios = list(df['smile_ratio'])
sm = [round(s, 3) for s in smile_ratios]
times = list(df['times'])
date_time = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in times]

plt.plot(date_time, sm)

plt.xlabel('Time')
plt.ylabel('Smile ratio')
plt.title('Smile detector')
plt.grid(True)
plt.show()
