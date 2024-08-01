# PySASF
A Python package for Source Apportionment with Sediment Fingerprinting
## Install
Download and extract...

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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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

First we need to import the `clarckeminella` analysis module. We will call it `cm`. 


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
    Done! Time for processing and save: 2.208845853805542


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


The Clarke and Minella's criterion for considering a feasible solution is that the proportion contributed by each source $P_i$ is such that $0<P_i<1$. We can extract the feaseble solutions usin a function `cm_feasebles` of `clarckeminella` analysis module. This is showed below.


```python
Pfea = cm.cm_feasebles(Ps)
print("The total number of feasible solution is:", len(Pfea))
```

    The total number of feasible solution is: 8132


A confidence region can be calculated in 2 dimentions using the $95 \%$ points closest to the feaseble proportions average usin distances given by $(P_i-P^*)^T S^{-1}(P_i-P^*)$, where $S$ is the $2 \times 2$ variance-covariance matrix of the feasible solutions and 
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


    
![png](output_23_0.png)
    





    <scipy.spatial._qhull.ConvexHull at 0x7f9522338c20>




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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>12.3411</td>
      <td>0.3506</td>
      <td>0.0433</td>
      <td>3240</td>
      <td>497</td>
      <td>0.289694</td>
      <td>0.171488</td>
      <td>0.538821</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>6.8760</td>
      <td>0.3801</td>
      <td>0.0261</td>
      <td>6480</td>
      <td>2071</td>
      <td>0.302395</td>
      <td>0.226502</td>
      <td>0.471103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>4.0042</td>
      <td>0.3924</td>
      <td>0.0157</td>
      <td>12960</td>
      <td>3384</td>
      <td>0.313042</td>
      <td>0.239722</td>
      <td>0.447235</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>2.5776</td>
      <td>0.3984</td>
      <td>0.0103</td>
      <td>19440</td>
      <td>3726</td>
      <td>0.341491</td>
      <td>0.245550</td>
      <td>0.412957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>1.6122</td>
      <td>0.3996</td>
      <td>0.0064</td>
      <td>25920</td>
      <td>5283</td>
      <td>0.343188</td>
      <td>0.247521</td>
      <td>0.409290</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20</td>
      <td>1.2406</td>
      <td>0.4008</td>
      <td>0.0050</td>
      <td>32400</td>
      <td>7238</td>
      <td>0.342542</td>
      <td>0.244117</td>
      <td>0.413339</td>
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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>13.4416</td>
      <td>0.3531</td>
      <td>0.0475</td>
      <td>3888</td>
      <td>623</td>
      <td>0.277980</td>
      <td>0.216130</td>
      <td>0.505891</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>7.8756</td>
      <td>0.3777</td>
      <td>0.0297</td>
      <td>7776</td>
      <td>1576</td>
      <td>0.352610</td>
      <td>0.246842</td>
      <td>0.400547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>4.2492</td>
      <td>0.3929</td>
      <td>0.0167</td>
      <td>15552</td>
      <td>3862</td>
      <td>0.343837</td>
      <td>0.259021</td>
      <td>0.397140</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>2.0408</td>
      <td>0.3988</td>
      <td>0.0081</td>
      <td>23328</td>
      <td>4551</td>
      <td>0.338644</td>
      <td>0.231789</td>
      <td>0.429568</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>1.1840</td>
      <td>0.4004</td>
      <td>0.0047</td>
      <td>31104</td>
      <td>6474</td>
      <td>0.336209</td>
      <td>0.233465</td>
      <td>0.430326</td>
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


    
![png](output_32_0.png)
    



```python

```


```python

```
