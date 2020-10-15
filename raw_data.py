{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "Python 3.7.0 64-bit ('mun': conda)",
   "display_name": "Python 3.7.0 64-bit ('mun': conda)",
   "metadata": {
    "interpreter": {
     "hash": "8378c1c5fd68f61f8c0782b63791a3ac8d651e4bfa45a9534f7a573cbbbb6dd4"
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import data_create_tool as dct\n",
    "import gzip\n",
    "import pickle\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.impute import KNNImputer\n",
    "from tqdm import tqdm_notebook\n",
    "from pyts.preprocessing import InterpolationImputer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "with gzip.open(\"./dataset/org_add_rain.pickle\" ,'rb') as f:\n",
    "    org_data = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "org_data = org_data.fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_by_area = org_data.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": "HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=8.0), HTML(value='')))",
      "application/vnd.jupyter.widget-view+json": {
       "version_major": 2,
       "version_minor": 0,
       "model_id": "9c5b03ae6bd64791baf6a505f35c3453"
      }
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "for col in tqdm_notebook(['SO2','NO2','NO','NOX','CO','O3','PM10','PM25']):\n",
    "    flag = col + '_FLAG'\n",
    "    data_by_area[flag][data_by_area[flag] == 2] = 0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": "HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=8.0), HTML(value='')))",
      "application/vnd.jupyter.widget-view+json": {
       "version_major": 2,
       "version_minor": 0,
       "model_id": "5d2a96210bcd4cbbbc8c58748101119c"
      }
     },
     "metadata": {}
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Series([], Name: SO2_FLAG, dtype: int64)\nSeries([], Name: NO2_FLAG, dtype: int64)\nSeries([], Name: NO_FLAG, dtype: int64)\nSeries([], Name: NOX_FLAG, dtype: int64)\nSeries([], Name: CO_FLAG, dtype: int64)\nSeries([], Name: O3_FLAG, dtype: int64)\nSeries([], Name: PM10_FLAG, dtype: int64)\nSeries([], Name: PM25_FLAG, dtype: int64)\n\n"
     ]
    }
   ],
   "source": [
    "for col in tqdm_notebook(['SO2','NO2','NO','NOX','CO','O3','PM10','PM25']):\n",
    "    flag = col + '_FLAG'\n",
    "    print(data_by_area[flag][data_by_area[flag] == 2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_by_area = data_by_area[['AREA_INDEX','MDATETIME','RAIN','SO2','NO2','NO','NOX','O3','CO','PM10','PM25','SO2_FLAG','NO2_FLAG','NO_FLAG','NOX_FLAG','O3_FLAG','CO_FLAG','PM10_FLAG','PM25_FLAG']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with gzip.open('./testset/raw_201015.pickle','wb')as f:\n",
    "    pickle.dump(data_by_area,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "         AREA_INDEX            MDATETIME     SO2     NO2     O3   CO  PM10  \\\n",
       "0            111121  2018/04/01 01:00:00  0.0050  0.0320  0.030  0.4  45.0   \n",
       "1            111122  2018/04/01 01:00:00  0.0050  0.0380  0.027  0.5  59.0   \n",
       "2            111123  2018/04/01 01:00:00  0.0070  0.0320  0.032  0.5  47.0   \n",
       "3            111124  2018/04/01 01:00:00  0.0060  0.0390  0.022  0.5  49.0   \n",
       "4            111125  2018/04/01 01:00:00  0.0040  0.0440  0.019  0.5  49.0   \n",
       "...             ...                  ...     ...     ...    ...  ...   ...   \n",
       "6386578      831155  2020/01/01 00:00:00  0.0060  0.0380  0.003  0.5  18.0   \n",
       "6386579      831481  2020/01/01 00:00:00  0.0018  0.0029  0.029  0.3  25.0   \n",
       "6386580      831491  2020/01/01 00:00:00  0.0008  0.0013  0.033  0.2  10.0   \n",
       "6386581      831492  2020/01/01 00:00:00  0.0008  0.0026  0.036  0.2  12.0   \n",
       "6386582      831493  2020/01/01 00:00:00  0.0030  0.0100  0.026  0.2  16.0   \n",
       "\n",
       "         PM25      NO     NOX  SO2_FLAG  NO2_FLAG  O3_FLAG  CO_FLAG  \\\n",
       "0        22.0  0.0010  0.0330         1         1        1        1   \n",
       "1        28.0  0.0130  0.0510         1         1        1        1   \n",
       "2        20.0  0.0020  0.0340         1         1        1        1   \n",
       "3        24.0  0.0070  0.0460         1         1        1        1   \n",
       "4        20.0  0.0040  0.0480         1         1        1        1   \n",
       "...       ...     ...     ...       ...       ...      ...      ...   \n",
       "6386578  18.0  0.0320  0.0700         1         1        1        1   \n",
       "6386579  17.0  0.0002  0.0031         1         1        1        1   \n",
       "6386580   9.0  0.0005  0.0018         1         1        1        1   \n",
       "6386581   9.0  0.0005  0.0030         1         1        1        1   \n",
       "6386582  12.0  0.0020  0.0120         1         1        1        1   \n",
       "\n",
       "         PM10_FLAG  PM25_FLAG  NO_FLAG  NOX_FLAG  RAIN  TIME_INDEX  \n",
       "0                1          1        1         1   0.0         0.0  \n",
       "1                1          1        1         1   0.0         0.0  \n",
       "2                1          1        1         1   0.0         0.0  \n",
       "3                1          3        1         1   0.0         0.0  \n",
       "4                1          1        1         1   0.0         0.0  \n",
       "...            ...        ...      ...       ...   ...         ...  \n",
       "6386578          1          1        1         1   0.0     15359.0  \n",
       "6386579          1          1        1         1   0.0     15359.0  \n",
       "6386580          1          1        1         1   0.0     15359.0  \n",
       "6386581          1          1        1         1   0.0     15359.0  \n",
       "6386582          1          3        1         1   0.0     15359.0  \n",
       "\n",
       "[6386583 rows x 20 columns]"
      ],
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>AREA_INDEX</th>\n      <th>MDATETIME</th>\n      <th>SO2</th>\n      <th>NO2</th>\n      <th>O3</th>\n      <th>CO</th>\n      <th>PM10</th>\n      <th>PM25</th>\n      <th>NO</th>\n      <th>NOX</th>\n      <th>SO2_FLAG</th>\n      <th>NO2_FLAG</th>\n      <th>O3_FLAG</th>\n      <th>CO_FLAG</th>\n      <th>PM10_FLAG</th>\n      <th>PM25_FLAG</th>\n      <th>NO_FLAG</th>\n      <th>NOX_FLAG</th>\n      <th>RAIN</th>\n      <th>TIME_INDEX</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>111121</td>\n      <td>2018/04/01 01:00:00</td>\n      <td>0.0050</td>\n      <td>0.0320</td>\n      <td>0.030</td>\n      <td>0.4</td>\n      <td>45.0</td>\n      <td>22.0</td>\n      <td>0.0010</td>\n      <td>0.0330</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>111122</td>\n      <td>2018/04/01 01:00:00</td>\n      <td>0.0050</td>\n      <td>0.0380</td>\n      <td>0.027</td>\n      <td>0.5</td>\n      <td>59.0</td>\n      <td>28.0</td>\n      <td>0.0130</td>\n      <td>0.0510</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>111123</td>\n      <td>2018/04/01 01:00:00</td>\n      <td>0.0070</td>\n      <td>0.0320</td>\n      <td>0.032</td>\n      <td>0.5</td>\n      <td>47.0</td>\n      <td>20.0</td>\n      <td>0.0020</td>\n      <td>0.0340</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>111124</td>\n      <td>2018/04/01 01:00:00</td>\n      <td>0.0060</td>\n      <td>0.0390</td>\n      <td>0.022</td>\n      <td>0.5</td>\n      <td>49.0</td>\n      <td>24.0</td>\n      <td>0.0070</td>\n      <td>0.0460</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>3</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>111125</td>\n      <td>2018/04/01 01:00:00</td>\n      <td>0.0040</td>\n      <td>0.0440</td>\n      <td>0.019</td>\n      <td>0.5</td>\n      <td>49.0</td>\n      <td>20.0</td>\n      <td>0.0040</td>\n      <td>0.0480</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>6386578</th>\n      <td>831155</td>\n      <td>2020/01/01 00:00:00</td>\n      <td>0.0060</td>\n      <td>0.0380</td>\n      <td>0.003</td>\n      <td>0.5</td>\n      <td>18.0</td>\n      <td>18.0</td>\n      <td>0.0320</td>\n      <td>0.0700</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>15359.0</td>\n    </tr>\n    <tr>\n      <th>6386579</th>\n      <td>831481</td>\n      <td>2020/01/01 00:00:00</td>\n      <td>0.0018</td>\n      <td>0.0029</td>\n      <td>0.029</td>\n      <td>0.3</td>\n      <td>25.0</td>\n      <td>17.0</td>\n      <td>0.0002</td>\n      <td>0.0031</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>15359.0</td>\n    </tr>\n    <tr>\n      <th>6386580</th>\n      <td>831491</td>\n      <td>2020/01/01 00:00:00</td>\n      <td>0.0008</td>\n      <td>0.0013</td>\n      <td>0.033</td>\n      <td>0.2</td>\n      <td>10.0</td>\n      <td>9.0</td>\n      <td>0.0005</td>\n      <td>0.0018</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>15359.0</td>\n    </tr>\n    <tr>\n      <th>6386581</th>\n      <td>831492</td>\n      <td>2020/01/01 00:00:00</td>\n      <td>0.0008</td>\n      <td>0.0026</td>\n      <td>0.036</td>\n      <td>0.2</td>\n      <td>12.0</td>\n      <td>9.0</td>\n      <td>0.0005</td>\n      <td>0.0030</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>15359.0</td>\n    </tr>\n    <tr>\n      <th>6386582</th>\n      <td>831493</td>\n      <td>2020/01/01 00:00:00</td>\n      <td>0.0030</td>\n      <td>0.0100</td>\n      <td>0.026</td>\n      <td>0.2</td>\n      <td>16.0</td>\n      <td>12.0</td>\n      <td>0.0020</td>\n      <td>0.0120</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>3</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0.0</td>\n      <td>15359.0</td>\n    </tr>\n  </tbody>\n</table>\n<p>6386583 rows × 20 columns</p>\n</div>"
     },
     "metadata": {},
     "execution_count": 4
    }
   ],
   "source": [
    "data_by_area = dct.load_area_split(org_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with gzip.open('./testset/raw_201015_by_area.pickle','wb')as f:\n",
    "    pickle.dump(data_by_area,f)"
   ]
  }
 ]
}