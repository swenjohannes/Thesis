# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:02:21 2022

@author: Swen
"""
import pandas as pd
import numpy as np

def generate_table(results, col1, col2, rownames):
    #rownames = ["T", r"\gamma"]

    #Set up row names -> inferred from first 2 columns of the table
    index_arrays = np.array([
        np.array(results[:, 0]).astype(int),
        np.array(results[:, 1]).astype(int),
    ])

    index_df = df = pd.DataFrame(
        index_arrays.T,
        columns= rownames
    )

    multi_index = pd.MultiIndex.from_frame(index_df)
    df = pd.DataFrame(results[:, 2:], index=multi_index)
    df

    #Set up colnames
    m = len(col1)
    n = len(col2)

    col1 = np.repeat(col1, n)
    col2 = np.tile(col2, m)
    cols = np.array([col1, col2])
    cols = cols.T.tolist()

    column_names =pd.DataFrame(cols, columns = ['',''])
    columns = pd.MultiIndex.from_frame(column_names)

    #Get table
    df = pd.DataFrame(results[:, 2:(m * n + 3)], columns=columns, index=multi_index)
    latex = df.to_latex(escape=False,
                        float_format=lambda x: '%.3f' % x if np.abs(x) > 0.001 else '%.2e' % x,
                        sparsify= True)
    return latex
