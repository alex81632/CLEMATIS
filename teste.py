import numpy as np
from datetime import datetime, timedelta

timeST = datetime(2023, 9, 24, 9, 30, 35)

time_extra = timeST + timedelta(minutes=1)

#printar a diferença entre os dois tempos em minutos
print(int((time_extra - timeST).total_seconds()/60))

i = 2

teste = 'abc123'

print(str(i) in teste)