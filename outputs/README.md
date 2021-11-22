# Model predictions

## `chr*.npy` (TODO)
 
The model returns predictions for every nucleotide on the transcript, saved as a `.npy` file. For each transcript, the array lists the transcript label, input sequence and model outputs. Predictions on each chromosome are obtained by using the other chromosomes as train and validation data.
 
<details> 
 
--- 
``` 
>>> results = np.load('results.npy', allow_pickle=True) 
>>> results[0] 
array(['>ENST00000410304', 
       array([3, 1, 2, 1, 0, 2, 2, 3, 2, 2, 0, 1, 0, 2, 2, 0, 2, 1, 2, 1, 3, 0, 
              0, 2, 0, 2, 3, 2, 2, 2, 0, 0, 1, 2, 1, 1, 3, 1, 2, 1, 3, 0, 1, 2, 
              1, 2, 0, 3, 0, 0, 3, 2, 1, 0, 0, 3, 2, 0, 3, 3, 3, 1, 2, 0, 3, 3, 
              2, 2, 1, 0, 3, 1, 1, 0, 3, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 3, 3, 0, 
              1, 3, 3, 3, 3, 3, 0, 0, 0, 1, 3])                                , 
       array([2.3891837e-09, 7.0824785e-07, 8.3791534e-09, 4.3269135e-09, 
              4.9220684e-08, 1.5315813e-10, 7.0196869e-08, 2.4103475e-10, 
              4.5873511e-10, 1.4299616e-10, 6.1071654e-09, 1.9664975e-08, 
              2.9255699e-07, 4.7719610e-08, 7.7600065e-10, 9.2305236e-10, 
              3.3297397e-07, 3.5771163e-07, 4.1942007e-05, 4.5123262e-08, 
              1.2450059e-09, 9.2165324e-11, 3.6457399e-09, 8.8559119e-08, 
              9.2133210e-05, 1.7473910e-09, 4.0608841e-09, 2.9064828e-12, 
              1.9478179e-08, 9.0584736e-12, 1.7068935e-05, 2.8910944e-07, 
              3.5740332e-08, 3.3406838e-10, 5.7711222e-08, 5.0289093e-09, 
              7.4243858e-12, 2.2184177e-09, 5.2881451e-06, 6.1195571e-10, 
              1.4648888e-10, 1.4948037e-07, 2.3879443e-07, 1.6367457e-08, 
              1.9375465e-08, 3.3595885e-08, 4.1618881e-10, 6.3614699e-12, 
              4.1953702e-10, 1.3611480e-08, 2.0185058e-09, 8.1397658e-08, 
              2.3339116e-07, 4.8850779e-08, 1.6549968e-12, 1.2499275e-11, 
              8.3455109e-10, 1.5468280e-12, 3.5863316e-08, 1.2135585e-09, 
              4.4234839e-14, 2.0041482e-11, 4.0546926e-09, 4.8796110e-12, 
              3.4575018e-13, 5.0659910e-10, 3.2857072e-13, 2.3365734e-09, 
              8.3198276e-10, 2.9397595e-10, 3.3731489e-08, 9.1637538e-11, 
              1.0781720e-09, 1.0790679e-11, 4.8457072e-10, 4.6192927e-10, 
              4.9371015e-12, 2.8158498e-13, 2.9590792e-09, 4.3507330e-07, 
              5.7654831e-10, 2.4951474e-09, 4.6289192e-12, 1.5421598e-02, 
              1.0270607e-11, 1.1841109e-09, 7.9038587e-10, 6.5511790e-10, 
              6.0892291e-13, 1.6157842e-11, 6.9130129e-10, 4.5778301e-11, 
              2.1682500e-03, 2.3315516e-09, 2.2578116e-11], dtype=float32)], 
      dtype=object) 
 
``` 
--- 
  
 </details> 



## `homo_sapiens_proteome.csv`

Additional information was curated for the top scoring positions of each chromosome. For each chromosome, the top `k*3` predictions are included, where k is the number of translation initiation sites annotated by the Ensembl annotations (v102).

<details>

| **Column name**    | **Definition**                                                                                                                          |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| tr\_idx            | Index of the transcript in the data array (data/GRCh38p13)                                                                              |
| pos\_idx           | Index of nucleotide position as given for a given transcript in the data array (data/GRCh38p13)                                         |
| tr\_ID             | Ensembl Identifier for the transcript                                                                                                   |
| pos\_on\_tr        | Nucleotide position on the transcript (1-based coordinate)                                                                              |
| output             | Model probability output                                                                                                                |
| target             | (boolean) Ensembl TIS Annotation                                                                                                        |
| gene\_ID           | Ensembl Identifier of the Gene                                                                                                          |
| strand             | Strand on which the gene is present                                                                                                     |
| en\_tr\_type       | GENCODE tags on transcript biotype (see https://www.gencodegenes.org/pages/biotypes.html)                                               |
| en\_tags           | GENCODE tags on TIS annotation (see https://www.gencodegenes.org/pages/tags.html)                                                       |
| en\_tr\_support    | Transcript support level (see www.ensembl.org/info/genome/genebuild/transcript\_quality\_tags.html)                                     |
| tr\_len            | Transcript length                                                                                                                       |
| output\_rank       | Rank of output w.r.t. all outputs on the chromosome, lower rank denotes higher model probability                                        |
| top\_k             | (boolean) The prediction is in the top k probabilities on the chromosome (k: \# Ensembl TIS annotations)                                |
| TIS\_on\_tr        | (boolean) Number of potential TISs on the transcript, assuming all positions listed in the table are positive                           |
| tr\_has\_target    | (boolean) Whether the transcript has a TIS as annotated by Ensembl (see target)                                                         |
| dist\_from\_target | Distance to the annotated TIS, when present                                                                                             |
| frame\_wrt\_target | Reading frame w.r.t. the annotated TIS, when present                                                                                    |
| prot               | Resulting protein sequence resulting from translation starting at this position                                                         |
| prot\_len          | Length of resulting protein sequence                                                                                                    |
| prot\_id           | Protein ID of resulting protein sequence (if match is found)                                                                            |
| prot\_id\_type     | Type of resulting protein (only for TISs annotated by Ensembl)                                                                          |
| is\_target\_prot   | (boolean) The resulting protein is or has an identical sequence to a protein annotated by Ensembl                                       |
| prot\_count        | The number of times this protein is present in this table.                                                                              |
| TIS\_loc           | TIS location w.r.t. annotated TIS (aTIS): aTIS, 5\_UTR, 3\_UTR, nontranslated region (NTR), in CDS of aTIS (in\_CDS)                    |
| R\_value\_Jurkat   | ~~R\_{\\text{LTM}} - R\_{\\text{CHX}}~~ values for PROTEOFORMER TIS calls on data: (https://www.ncbi.nlm.nih.gov/sra?term=SRP065022)    |
| R\_value\_HCT116   | ~~R\_{\\text{LTM}} - R\_{\\text{CHX}}~~ values for PROTEOFORMER TIS calls on data: (https://www.ncbi.nlm.nih.gov/sra/?term=SRP042937)   |
| chrom              | chromosome                                                                                                                              |
| bl\_uniprot\_id    | Uniprot ID from top BLAST search hit                                                                                                    |
| bl\_perc\_score    | Percentage overlap score returned by top BLAST search hit                                                                               |
| bl\_e\_val         | E value of top BLAST search hit                                                                                                         |

</details>
