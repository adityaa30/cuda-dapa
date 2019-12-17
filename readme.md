# Design & Analysis of Parallel Algorithms - Assignment

## Problem Statement

Given a matrix of n*n elements, it is required to find the transpose of the given matrix. This can be solved using many architectures. This repository shows the use of Shuffle and Mesh Transpose methods to do the same.

## Assumptions

1. Each processor P(i,j) in Mesh Transpose holds data element a<sub>ij</sub>, and at the end of computation P(i,j) should hold a<sub>ji</sub>.
2. A processor will have the same performance irrespective of the architecture it is used in.  
3. In the case of shuffle transpose value of n=2<sup>q</sup> where n is the order of the given matrix.

## Methods

**1. Transpose using Mesh Architecture** 
processor P(i,j) has three registers:

1. A(i,j) is used to store a<sub>ij</sub> initially and a<sub>ji</sub> when the algorithm terminates.
2. B(i,j) is used to store data received from P(i,j + 1) or P(i - 1,j), that is, from its right or top neighbours.
3. C(i,j) is used to store data received from P(i,j - 1) or P(i + 1,j), that is, from its left or bottom neighbours.

Implementation: [transpose_mesh.cu](https://github.com/adityaa30/cuda-dapa/blob/master/transpose_mesh.cu)

Example Output:
![Mesh Transpose Example](https://github.com/adityaa30/cuda-dapa/blob/master/mesh.png "Mesh Transpose Example")

**2.  Transpose using Perfect Shuffle**
After q shuffles (i.e., q cyclic shifts to the left), the element originally held by P<sub>k</sub> will be in the processor whose index is 2<sup>q</sup>(j-1) + (i-1).

Implementation: [transpose_shuffle.cu](https://github.com/adityaa30/cuda-dapa/blob/master/transpose_shuffle.cu)

## Analysis

1. **Sequential Transpose**
    procedure runs in O(n<sup>2</sup>) time, which is optimal in view of the Ω(n<sup>2</sup>) steps required to simply read A.
2. **Transpose using Mesh Architecture**  
    Each element a<sub>ij</sub>, i > j, must travel up its column until it reaches P(j, j) and then travel along a row until it settles in P(j, i). Similarly for a<sub>ij</sub>, j > i. The longest path is the one traversed by a<sub>n1</sub>, (or a<sub>1n</sub>), which consists of 2(n -1) steps. The running time is therefore  `t(n) = O(n)` which is the best possible for the mesh. Since p(n) = n<sup>2</sup> , the procedure has a cost of O(n<sup>3</sup>), which is not optimal.  
3. **Transpose using Perfect Shuffle**  
    There are q constant time iterations and therefore the procedure runs in t(n) = O(log n) time. Since p(n) = n<sup>2</sup> , c(n) = O(n<sup>2</sup> log n), which is not optimal. Interestingly, the shuffle interconnection is faster than the mesh in computing the transpose of a matrix. This is contrary to our original intuition, which suggested that the mesh is the most naturally suited geometry for matrix operations.

## References

1. [Aki, S.G.  _The design and analysis of parallel algorithms_. United States: N. p., 1989. Web](https://www.osti.gov/biblio/5692994)