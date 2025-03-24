import streamlit as st
import pandas as pd
from scipy.spatial import KDTree

# Data in dictionary format
data = {
    'MMSE': [29, 27, 29, 29, 30, 28, 29, 28, 27, 30, 30, 30, 29, 29, 29, 28, 30, 30, 30, 30, 28, 27, 29, 30, 30, 30, 30, 30, 28, 28, 27, 23, 29, 30, 30, 30, 30, 30, 28, 28, 27, 29, 30, 30, 30, 30, 30, 29, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 28, 27, 30, 28, 27, 29, 28, 27, 29, 30, 30, 30, 30, 29, 29, 29, 29, 28, 30, 30, 30, 30, 29, 29, 29, 28],
    'eTIV': [1432, 1432, 1548, 1534, 1550, 1511, 1506, 1383, 1390, 1513, 1449, 1769, 1785, 1814, 1154, 1165, 1611, 1628, 1446, 1412, 1475, 1484, 1583, 1586, 1590, 1548, 1619, 1443, 1479, 1507, 1429, 1502, 1491, 1554, 1420, 1452, 1686, 1784, 1569, 1575, 1425, 1688, 1564, 1551, 1550, 1556, 1753, 1777, 1560, 1781, 1472, 1520, 1444, 1517, 1561, 1477, 1579, 1778, 1566, 1441, 1469, 1505, 1522, 1526, 1603, 1481, 1467, 1503, 1568, 1495, 1478, 1576, 1558, 1442, 1490, 1509, 1553, 1549, 1482, 1779, 1562, 1508, 1557, 1473, 1431, 1571],
    'nWBV': [0.692, 0.684, 0.773, 0.772, 0.758, 0.739, 0.715, 0.748, 0.728, 0.771, 0.774, 0.699, 0.687, 0.679, 0.75, 0.736, 0.729, 0.709, 0.78, 0.783, 0.762, 0.75, 0.777, 0.757, 0.76, 0.733, 0.727, 0.748, 0.772, 0.773, 0.764, 0.762, 0.768, 0.752, 0.735, 0.736, 0.685, 0.688, 0.713, 0.711, 0.703, 0.685, 0.731, 0.759, 0.752, 0.74, 0.703, 0.704, 0.741, 0.695, 0.719, 0.728, 0.71, 0.716, 0.737, 0.701, 0.711, 0.706, 0.724, 0.743, 0.745, 0.738, 0.73, 0.74, 0.718, 0.757, 0.762, 0.715, 0.724, 0.739, 0.758, 0.774, 0.75, 0.747, 0.727, 0.755, 0.752, 0.727, 0.712, 0.702, 0.745, 0.739, 0.718, 0.722],
    'ASF': [1.225, 1.226, 1.134, 1.144, 1.133, 1.162, 1.166, 1.269, 1.263, 1.16, 1.212, 0.992, 0.983, 0.967, 1.521, 1.506, 1.089, 1.078, 1.214, 1.243, 1.19, 1.183, 1.108, 1.107, 1.104, 1.131, 1.119, 1.211, 1.134, 1.12, 1.148, 1.156, 1.173, 1.165, 1.212, 1.207, 1.013, 1.015, 1.176, 1.171, 1.186, 1.009, 1.196, 1.156, 1.179, 1.184, 1.005, 1.002, 1.174, 1.01, 1.198, 1.205, 1.199, 1.193, 1.177, 1.01, 1.181, 1.195, 1.181, 1.179, 1.15, 1.149, 1.168, 1.191, 1.18, 1.096, 1.095, 1.168, 1.16, 1.15, 1.171, 1.153, 1.181, 1.142, 1.12, 1.137, 1.147, 1.125, 1.144, 1.145, 1.16, 1.148, 1.145, 1.124, 1.151],
    'Group': ['Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Converted', 'Converted', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented']
}

# Create DataFrame from data
df = pd.DataFrame(data)

# Create KDTree for nearest neighbor search
features = df[['MMSE', 'eTIV', 'nWBV', 'ASF']].values
kdtree = KDTree(features)

# Streamlit App
def main():
    st.title("Dementia Prediction System")
    
    # User inputs
    mmse = st.number_input("Enter MMSE value:", min_value=18, max_value=30, value=28)
    etiv = st.number_input("Enter eTIV value:", min_value=1100, max_value=1850, value=1500)
    nwbv = st.number_input("Enter nWBV value:", min_value=0.65, max_value=0.80, value=0.74, step=0.001)
    asf = st.number_input("Enter ASF value:", min_value=0.95, max_value=1.6, value=1.2, step=0.001)
    
    # Button to predict
    if st.button("Predict Group"):
        # Find nearest neighbor
        _, idx = kdtree.query([mmse, etiv, nwbv, asf])
        
        # Get nearest group
        group = df.iloc[idx]['Group']
        st.success(f"The predicted group is: **{group}**")

if __name__ == "__main__":
    main()
