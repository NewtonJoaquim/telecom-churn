#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = read_csv('data-projeto-telecom/projeto4_telecom_treino.csv')


# In[3]:


data.shape


# In[4]:


data.head()


# #### Vamos checar a correlação entre as variaveis

# In[5]:


corrMatrix = data.corr(method='pearson')


# #### Vamos remover a coluna de ids

# In[6]:


data = data.drop(data.columns[0], axis =1)


# #### Vamos transformar as variáveis categóricas. Temos as variáveis state, area_code, international_plan, voicemail_plan, churn

# In[7]:


from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
for feature in ['state', 'area_code', 'international_plan', 'voice_mail_plan', 'churn']:
    LE.fit(data[feature])
    data[feature] = LE.transform(data[feature])


# In[8]:


data.head()


# #### Visualizações

# In[9]:


data.hist()
plt.show()


# In[10]:


# Matriz de Correlação genérica
correlations = data.corr()

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
plt.show()


# In[11]:


correlations


# #### Podemos observar que as colunas total day minutes e total day charge tem correlação 1, e, de fato, o valor cobrado depende apenas do numero de minutos falado. Podemos remover uma destas colunas

# In[12]:


data = data.drop('total_day_minutes', 1)


# #### Da mesma forma, removemos a coluna total_night_minutes, total_intl_minutes e total_eve_minutes

# In[13]:


data=data.drop('total_night_minutes', 1)


# In[14]:


data = data.drop('total_intl_minutes', 1)


# In[15]:


data = data.drop('total_eve_minutes', 1)


# #### Ainda existem colunas com correlação acima de 0.9. Removeremos estas também para evitar de vez o problema da multicolinearidade. Iremos eliminar a coluna number_vmail_messages

# In[16]:


data = data.drop('number_vmail_messages', 1)


# #### Conseguimos eliminar a multicolinearidade

# In[17]:


# Matriz de Correlação genérica
correlations = data.corr()

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
plt.show()


# #### Agora devemos normalizar os dados. Como os dados não são muito esparsos, optaremos por utilizar o MinMaxScaler

# In[18]:


from sklearn.preprocessing import MinMaxScaler
X = data.values[:, 0:-1]
Y = data.values[:, -1]
scaler = MinMaxScaler(feature_range=(0,1))
normalizedX = scaler.fit_transform(X)


# In[19]:


normalizedX.shape


# #### Dividimos os dados em treino e teste

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(normalizedX, Y, test_size=0.33)


# #### Agora vamos treinar o modelo

# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)


# In[22]:


model.score(X_test, Y_test)


# #### Agora vmos inserir as predições do arquivo de testes em um arquivo de resultados. Para isso, precisamos fazer o pré-processamento nestes dados

# In[42]:


test_file = 'data-projeto-telecom/projeto4_telecom_teste.csv'
test_data = read_csv(test_file)
test_data_ids = test_data.iloc[:,0]
test_data = test_data.drop(test_data.columns[0], axis =1)
test_data.columns

#LE=LabelEncoder()
for feature in ['state', 'area_code', 'international_plan', 'voice_mail_plan', 'churn']:
    LE.fit(test_data[feature])
    test_data[feature] = LE.transform(test_data[feature])
    
test_data = test_data.drop('total_day_minutes', 1)
test_data = test_data.drop('total_night_minutes', 1)
test_data = test_data.drop('total_intl_minutes', 1)
test_data = test_data.drop('total_eve_minutes', 1)
test_data = test_data.drop('number_vmail_messages', 1)

test_X = test_data.values[:, 0:-1]
test_Y = test_data.values[:, -1]
scaler = MinMaxScaler(feature_range=(0,1))
normalized_test_X = scaler.fit_transform(test_X)


# In[35]:


prediction = model.predict(normalized_test_X)


# In[36]:


prediction


# In[44]:


f = open("churn_submission.csv", "w+")
f.write("ID,TARGET\n")
for i in range(len(prediction)):
    f.write(str(test_data_ids.values[i]) + "," + str(int(prediction[i])) + "\n")
f.close()


# In[ ]:




