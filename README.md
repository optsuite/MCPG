## MCPG 

**MCPG** is an efficient and stable framework for solving Binary Optimization problems based on a **M**onte **C**arlo **P**olicy **G**radient Method with Local Search:  
$$\min_x \quad f(x), \quad\mathrm{s.t.}\quad x_i \in \\{1,-1\\}.$$

## Algorithm
MCPG  consists of the following main components:

* a filter function $T(x)$ that enhances the objective function, reducing the probability of the algorithm from falling into local minima;

* a sampling procedure with filter function, which starts from the best solution found in previous steps and tries to keep diversity; 

* a modified policy gradient algorithm to update the probabilistic model;

* a probabilistic model that outputs a distribution $p_\theta(\cdot|\mathcal{P})$, guiding the sampling procedure towards potentially good solutions.

The pipeline of MCPG is demonstrated in the next figure. In each iteration, MCPG starts from the best samples of the previous iteration and performs MCMC sampling in parallel. The algorithm strives to obtain the best solutions with the aid of the powerful probabilistic model. To improve the efficiency, a filter function is applied to compute a modified objective function. At the end of the iteration, the probabilistic model is updated using policy gradient, ensuring to push the boundaries of what is possible in the quest for optimal solutions. 

![algo](algo.png)


## Code Structure

- mcpg.py: Our MCPG solver.
- model.py: The probabilistic model to output the mean-field distribution.
- dataloader.py: Data loader for MCPG to input the problem instance.
- sampling.py: The sampling procedure combining with the local search algorithm in MCPG.

## Examples

### maxcut
The MaxCut problem aims to divide a given weighted graph $G = (V,E)$ into two parts and maximize the total weight of the edges connecting two parts. This problem can be expressed as a binary programming problem:
$$\max  \quad  \sum_{(i,j) \in E} w_{ij} (1-x_i x_j), \quad \mathrm{s.t.}\quad  x\in \\{-1, 1\\}^n.$$

For solving maxcut problem using MCPG, run the following code

```
python mcpg.py config/maxcut_default.yaml data/graph/G14.txt
```

### Quadratic Unconstrained Binary Optimization
QUBO is to solve the following problem:
$$\max \quad  x^\mathrm{T} Q x,\quad\mathrm{s.t.}\quad x\in \\{0, 1\\}^n.$$
The sparsity of $Q$ in our experiments is greater than $0.5$, which fundamentally differs from instances derived from graphs such as Gset dataset, where the sparsity is less than $0.1$.

For solving QUBO problem using MCPG, run the following code
```python
python mcpg.py config/qubo_default.yaml data/nbiq/nbiq_5000_1.npy
```

### Cheeger Cut
Cheeger cut is a kind of balanced graph cut, which are widely used in classification tasks and clustering. Given a graph $G = (V, E, w)$, the ratio Cheeger cut (RCC) and the normal Cheeger cut (NCC) are defined as
$$\mathrm{RCC}(S, S^c)  = \frac{\mathrm{cut}(S,S^c)}{\min\\{|S|, |S^c|\\}},\quad\mathrm{NCC}(S, S^c)  = \frac{\mathrm{cut}(S,S^c)}{|S|} + \frac{\mathrm{cut}(S,S^c)}{|S^c|},$$
where $S$ is a subset of $V$ and $S^c$ is its complementary set. The task is to find the minimal ratio Cheeger cut or normal Cheeger cut, which can be converted into the following binary unconstrained programming:
$$\min \quad \frac{\sum_{(i,j)\in E}(1-x_ix_j)}{\min \sum_{i=1:n} (1 + x_i), \sum_{i=1:n} (1 - x_i)},\quad \mathrm{s.t.} \quad x\in\\{-1,1\\}^n,$$ 
and 
$$\min\quad \frac{\sum_{(i,j)\in E}(1-x_ix_j)}{\sum_{i=1:n} (1 + x_i)} + \frac{\sum_{(i,j)\in E}(1-x_ix_j)}{\sum_{i=1:n} (1 - x_i)},\quad \mathrm{s.t.} \quad x\in\\{-1,1\\}^n.$$

For solving the Cheeger cut problem using MCPG, run the following code
```python
python mcpg.py config/rcheegercut_default.yaml data/graph/G14.txt
python mcpg.py config/ncheegercut_default.yaml data/graph/G14.txt
```

### MIMO
The MIMO problem is to recover $x_C \in \mathcal Q$ from the linear model
$$y_C = H_Cx_C+\nu_C,$$
where $y_C\in \mathbb C^M$ denotes the received signal, $H_C\in \mathbb C^{M\times N}$ is the channel, $x_C$ denotes the sending signal, and $\nu_C\in \mathbb C^N\sim \mathcal N(0,\sigma^2I_N)$ is the Gaussian noise with known variance. 
 
The problem can be reduced to a binary one and is equivalent to the following: 
$$\min_{x\in\mathbb{R}^{2N}}\quad\|Hx-y\|_2^2,\quad\mathrm{s.t.} \quad x\in \\{-1, 1\\}^{2N}.$$

For solving the MIMO problem using MCPG, run the following code
```python
python mcpg.py config/mimo_default.yaml data/mimo/~.mat
```

### MaxSAT
The MaxSAT problem is to find an assignment of the variables that satisfies the maximum number of clauses in a boolean formula in conjunctive normal form (CNF). Given a formula in CNF consists of clause $c^1,c^2,\cdots,c^m$, we formulate the partial MaxSAT problem as a penalized binary programming problem:
$$\max \quad \sum_{c^i \in C_1\cup C_2} w_i\max\\{c_1^i x_1, c_2^i x_2,\cdots, c_n^i x_n , 0\\},\quad \text{s.t.} \quad  x \in \\{-1,1\\}^n,$$
where $w_i = 1$ for $c^i \in C_1$ and $w_i = |C_1| + 1$ for $c^i \in C_2$. $C_1$ represents the soft clauses that should be satisfied as much as possible and $C_2$ represents the hard clauses that have to be satisfied. 

$c_j^i$ represents the sign of literal $j$ in clause $i$. $c_j^i = 1$ when $x_j$ appears in the clause $C_i$, $c_j^i = -1$ when $\neg x_j$ appears in the clause $C_i$ and  otherwise $c_j^i = 0$.

The constraints in the partial MaxSAT problem can be converted to an exact penalty in the objective function, which is demonstrated in \eqref{eq:penaltyfun}. Since the left side of the equality constraints in \eqref{eq:maxsat} is no more than 1, the absolute function of the penalty can be dropped. Therefore, we have the following binary programming problem:

For solving the maxsat problem using MCPG, run the following code
```python
python mcpg.py config/maxsat_default.yaml data/sat/randu_1.cnf
```
For solving the partial maxsat problem using MCPG, run the following code
```python
python mcpg.py config/pmaxsat_default.yaml data/sat/randu_1.cnf
```

## Summary of Datasets 

We list the downloading links to the datasets used in the papers for reproduction.

* Gset instance: http://www.stanford.edu/yyye/yyye/Gset
* Generated large regular graph datasets: http://faculty.bicmr.pku.edu.cn/~wenzw/code/regular_graph.zip
* Generated NBIQ datasets: http://faculty.bicmr.pku.edu.cn/~wenzw/code/nbiq.zip
* Generated MaxSAT datasets: http://faculty.bicmr.pku.edu.cn/~wenzw/code/maxsat.zip
* Max-SAT Evaluation 2016: [Max-SAT 2016 - Eleventh Max-SAT Evaluation (udl.cat)](http://maxsat.ia.udl.cat/benchmarks/)
* MIMO Simulation: http://faculty.bicmr.pku.edu.cn/~wenzw/code/mimo.zip 

## Contact 

We hope that the package is useful for your application. If you have any bug reports or comments, please feel free to email one of the toolbox authors:

- Cheng Chen, chen1999 at pku.edu.cn
- Ruitao Chen, chenruitao at stu.pku.edu.cn
- Tianyou Li, tianyouli at stu.pku.edu.cn
- Zaiwen Wen, wenzw at pku.edu.cn

## Reference

## License
GNU General Public License v3.0
