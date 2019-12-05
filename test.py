from pandas import DataFrame
import matplotlib.pyplot as plt

Data = {'Year': [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010],
        'Unemployment_Rate': [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]
        }

df = DataFrame(Data, columns=['Year', 'Unemployment_Rate'])
df.plot(x='Year', y='Unemployment_Rate', kind='line')
plt.show()