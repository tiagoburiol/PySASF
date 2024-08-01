<img src="https://github.com/tiagoburiol/PySASF/blob/main/images/logo.png?raw=true" style="height: 80px; width:80px;"/>


# PySASF
**A Python package for Source Apportionment with Sediment Fingerprinting.**

PySASF has been developed to provide computational support for research aimed at identifying the percentages contributed by sources in fluvial sediments. The initiative originated from a collaboration between the Department of Soil Science and the Department of Mathematics at the Federal University of Santa Maria, with participation from other educational and research institutions. The initial motivation was to reproduce the results published in Clarke and Minella (2016) and to provide a package of Python routines so that the experiment can be repeated using other data sources.

## Install
Download from [here](https://github.com/tiagoburiol/PySASF/archive/refs/heads/main.zip), unzip and go to pysasf directory. Open `quick_star.ipynb` using [Jupyter Notebook](https://jupyter.org/) or [Jupyter Lab](https://jupyter.org/).

You will needs [NumPy](https://numpy.org/), [Scipy](https://scipy.org/), [MatplotLib](https://matplotlib.org/) instaled. All dependencies can be satisfied by an [Anaconda](https://anaconda.org/) installation.



## Example of usage
### 1. Loading the data

A good begin is to import the object class `BasinData` to instance and store data from a basin sediment sources. A instance of `BasinData` should be created and data load from a file. Is usual store data files in the 'data' directory one level above. The import and the creation of a instance of `BasinData` is showed below.


```python
from basindata import BasinData
```


```python
arvorezinha = BasinData("../data/arvorezinha_database.xlsx")
```

Since the file is load, some information and statistics can be visualizated as in the following examples .


```python
arvorezinha.infos()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sample Sizes</th>
      <th>Fe</th>
      <th>Mn</th>
      <th>Cu</th>
      <th>Zn</th>
      <th>Ca</th>
      <th>K</th>
      <th>P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>E</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>L</th>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
arvorezinha.means()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Means</th>
      <th>Fe</th>
      <th>Mn</th>
      <th>Cu</th>
      <th>Zn</th>
      <th>Ca</th>
      <th>K</th>
      <th>P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>6.21</td>
      <td>1470.45</td>
      <td>18.23</td>
      <td>79.71</td>
      <td>165.23</td>
      <td>3885.12</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>E</th>
      <td>6.76</td>
      <td>811.95</td>
      <td>23.28</td>
      <td>86.02</td>
      <td>76.10</td>
      <td>3182.27</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>L</th>
      <td>6.63</td>
      <td>1854.05</td>
      <td>20.05</td>
      <td>88.28</td>
      <td>159.17</td>
      <td>6572.31</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>6.16</td>
      <td>1119.02</td>
      <td>30.92</td>
      <td>99.66</td>
      <td>276.47</td>
      <td>9445.76</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>



```python
arvorezinha.std()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>STD</th>
      <th>Fe</th>
      <th>Mn</th>
      <th>Cu</th>
      <th>Zn</th>
      <th>Ca</th>
      <th>K</th>
      <th>P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>0.48</td>
      <td>548.49</td>
      <td>2.41</td>
      <td>7.84</td>
      <td>82.19</td>
      <td>1598.45</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.98</td>
      <td>399.90</td>
      <td>1.98</td>
      <td>6.96</td>
      <td>26.21</td>
      <td>948.95</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>L</th>
      <td>1.07</td>
      <td>399.77</td>
      <td>3.86</td>
      <td>15.70</td>
      <td>79.33</td>
      <td>2205.99</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>1.01</td>
      <td>294.13</td>
      <td>10.13</td>
      <td>8.40</td>
      <td>79.37</td>
      <td>2419.21</td>
      <td>0.02</td>
    </tr>
  </tbody>
</table>
</div>


### 2. Using the clarkeminela module

We can easily reproduce the Clarke and Minella (2016) method of measures the increase in uncertainty when sampling sediment fingerprinting. The full explanation of this method is avaliable in the paper "Evaluating sampling efficiency when estimating sediment source contributions to suspended sediment in rivers by fingerprinting". DOI: 10.1002/hyp.10866. The steps needed to achieve the same results described in paper can be executed by a few functions calls as will be shown below

First we need to import the `clarckeminella` analysis module. 


```python
import clarkeminella as cm
```

Now we will calculate and save in a file all the combinations possible for proportions contribuited by the sediment sources. The rotine `calculate_and_save_all_proportions()` will create two files, one for all cobinations possible for eath sample in data base, savis its indexes, and the corresponding proportions. The defoult method for calculate is the ordinary least square. Other methods can be choosed by `bd.set_solver_option(option)`. 


```python
arvorezinha.calculate_and_save_all_proportions(load=False)
```

    Calculating all proportions...
    Total combinations: 38880 , shape of proportions: (38880, 3)
    Saving combinations indexes in: C9E9L20Y24_combs.txt
    Saving proportions calculated in: C9E9L20Y24_props.txt
    Done! Time for processing and save: 2.0073955059051514


If you want to store the proportions solutions and the combination indexes, you can choose `load=True`(is the defoult option) when call the rotine above. The proportions solutions and the combination indexes wil be  stored on `BasinData`object class.

For read the files created and load proportions solutions and the combination indexes we can use the `load_combs_and_props_from_files(combs_file, props_file)` function. A example is showed below.


```python
combs, Ps = arvorezinha.load_combs_and_props_from_files('C9E9L20Y24_combs.txt',
                                                        'C9E9L20Y24_props.txt')
```

We can verify de array data loaded making.


```python
display(combs, Ps)
```


    array([[ 0,  0,  0,  0],
           [ 0,  0,  0,  1],
           [ 0,  0,  0,  2],
           ...,
           [ 8,  8, 19, 21],
           [ 8,  8, 19, 22],
           [ 8,  8, 19, 23]])



    array([[ 0.445 , -0.2977,  0.8526],
           [ 0.3761,  0.128 ,  0.4959],
           [ 0.3454,  0.1248,  0.5298],
           ...,
           [ 0.4963, -0.0081,  0.5118],
           [ 0.4212, -0.6676,  1.2464],
           [-0.0679, -0.138 ,  1.206 ]])


The Clarke and Minella's criterion for considering a feasible solution is that the proportion contributed by each source $P_i$ is such that $0 &lt P_i &lt 1$. We can extract the feaseble solutions usin a function `cm_feasebles` of `clarckeminella` analysis module. This is showed below.


```python
Pfea = cm.cm_feasebles(Ps)
print("The total number of feasible solution is:", len(Pfea))
```

    The total number of feasible solution is: 8132


A confidence region can be calculated in 2 dimentions using the $95 \%$ points closest to the feaseble proportions average using distances given by 
```math
(P_i-P^*)^T S^{-1}(P_i-P^*)
```
, where $S$ is the $2 \times 2$ variance-covariance matrix of the feasible solutions and 
$P^*$ is the mean of feaseble proportions.

A more detailed explanation can be can be obtained in the Clarke and Minella's paper.

The `clarckeminella` module  implement a function for get a confidence region, as can be seen in the example below.


```python
Pcr = cm.confidence_region(Pfea)
print("The total number of points in 95% confidence region is:", len(Pfea))
```

    The total number of points in 95% confidence region is: 8132


Lets draw the confidence region usin the `draw_hull(pts)` function.


```python
cm.draw_hull(Pcr)
```
![png](https://github.com/tiagoburiol/PySASF/blob/main/images/confidence_region.png)
    
```python
arvorezinha.props
```
    array([[ 0.445 , -0.2977,  0.8526],
           [ 0.3761,  0.128 ,  0.4959],
           [ 0.3454,  0.1248,  0.5298],
           ...,
           [ 0.4963, -0.0081,  0.5118],
           [ 0.4212, -0.6676,  1.2464],
           [-0.0679, -0.138 ,  1.206 ]])
```python
arvorezinha.combs
```
    array([[ 0,  0,  0,  0],
           [ 0,  0,  0,  1],
           [ 0,  0,  0,  2],
           ...,
           [ 8,  8, 19, 21],
           [ 8,  8, 19, 22],
           [ 8,  8, 19, 23]])

### 3. Processing data from reductions and repetitions 

As a result of Clarke and Minella's article presents 
table and graphs of average values ​​for 50 repetitions taking
subsamples of different sizes drawn from each sample set.
A 95% confidence regions are calculated for each sample reduction and the proportions $P_1$ and $P_2$,
along with the standard deviations is calculated.

De full analysis can be repreduced and customized usin the routine `run_repetitions_and_reduction (basindata, source_key, list_of_reductions,repetitions=50)`. The results is saved in a `csv`file an can be stored and load later. A example is showed below.


```python
tableY = cm.run_repetitions_and_reduction (arvorezinha, 'Y',[2,4,8,12,16,20,24])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nSamp</th>
      <th>CV</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Total</th>
      <th>Feas</th>
      <th>MeanP1</th>
      <th>MeanP2</th>
      <th>MeanP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>12.6361</td>
      <td>0.3588</td>
      <td>0.0453</td>
      <td>3240</td>
      <td>890</td>
      <td>0.320417</td>
      <td>0.335431</td>
      <td>0.344152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>6.4494</td>
      <td>0.3843</td>
      <td>0.0248</td>
      <td>6480</td>
      <td>1393</td>
      <td>0.338388</td>
      <td>0.194777</td>
      <td>0.466836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>4.0860</td>
      <td>0.3926</td>
      <td>0.0160</td>
      <td>12960</td>
      <td>3397</td>
      <td>0.360820</td>
      <td>0.236718</td>
      <td>0.402460</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>2.8462</td>
      <td>0.3948</td>
      <td>0.0112</td>
      <td>19440</td>
      <td>2907</td>
      <td>0.326356</td>
      <td>0.236953</td>
      <td>0.436690</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>1.7210</td>
      <td>0.3997</td>
      <td>0.0069</td>
      <td>25920</td>
      <td>5538</td>
      <td>0.340522</td>
      <td>0.230913</td>
      <td>0.428564</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20</td>
      <td>1.2673</td>
      <td>0.4010</td>
      <td>0.0051</td>
      <td>32400</td>
      <td>6627</td>
      <td>0.351153</td>
      <td>0.248147</td>
      <td>0.400700</td>
    </tr>
    <tr>
      <th>6</th>
      <td>24</td>
      <td>0.0000</td>
      <td>0.4024</td>
      <td>0.0000</td>
      <td>38880</td>
      <td>8132</td>
      <td>0.339917</td>
      <td>0.245394</td>
      <td>0.414688</td>
    </tr>
  </tbody>
</table>
</div>


    Saving in C9E9L20Y24_Y-2-4-8-12-16-20-24.csv



```python
tableL = cm.run_repetitions_and_reduction (arvorezinha, 'L',[2,4,8,12,16,20,])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nSamp</th>
      <th>CV</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Total</th>
      <th>Feas</th>
      <th>MeanP1</th>
      <th>MeanP2</th>
      <th>MeanP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>10.1205</td>
      <td>0.3609</td>
      <td>0.0365</td>
      <td>3888</td>
      <td>947</td>
      <td>0.321718</td>
      <td>0.227048</td>
      <td>0.451236</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>7.2572</td>
      <td>0.3730</td>
      <td>0.0271</td>
      <td>7776</td>
      <td>1602</td>
      <td>0.345648</td>
      <td>0.245737</td>
      <td>0.408615</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>3.7313</td>
      <td>0.3932</td>
      <td>0.0147</td>
      <td>15552</td>
      <td>3501</td>
      <td>0.355173</td>
      <td>0.258425</td>
      <td>0.386401</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>1.7429</td>
      <td>0.4017</td>
      <td>0.0070</td>
      <td>23328</td>
      <td>5154</td>
      <td>0.354780</td>
      <td>0.254139</td>
      <td>0.391080</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>1.3499</td>
      <td>0.4005</td>
      <td>0.0054</td>
      <td>31104</td>
      <td>6622</td>
      <td>0.331247</td>
      <td>0.242668</td>
      <td>0.426085</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20</td>
      <td>0.0000</td>
      <td>0.4024</td>
      <td>0.0000</td>
      <td>38880</td>
      <td>8132</td>
      <td>0.339917</td>
      <td>0.245394</td>
      <td>0.414688</td>
    </tr>
  </tbody>
</table>
</div>


    Saving in C9E9L20Y24_L-2-4-8-12-16-20.csv


Finally the results can be ploted by columns setting the files and the names of columns to be ploted, like the example below.



```python
import plots
files = ['C9E9L20Y24_Y-2-4-8-12-16-20-24.csv',
         'C9E9L20Y24_L-2-4-8-12-16-20.csv']

plots.plot_cm_outputs(files, 'nSamp', 'CV')
```


    
![png](https://github.com/tiagoburiol/PySASF/blob/main/images/cv_plot.png?raw=true)
    



```python

```


```python

```
