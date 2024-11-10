import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from skimage import io
import os

# Parameters
## SwinFusion
# WINDOW_SIZE = (64, 64) # Patch size
WINDOW_SIZE = (256, 256) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "./ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

MODEL = 'UNetformer'
# MODEL = 'FTUNetformer'
# MODEL = 'ABCNet'
# MODEL = 'CMTFNet'

MODE = 'Train'
# MODE = 'Test'
DATASET = 'Vaihingen'
# DATASET = 'Urban'
# LOSS = 'SEG'
# LOSS = 'SEG+BDY'
# LOSS = 'SEG+OBJ'
LOSS = 'SEG+BDY+OBJ'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(MODEL + ', ' + MODE + ', ' + DATASET + ', ' + LOSS)

if DATASET == 'Vaihingen':
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    Stride_Size = 32
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    BOUNDARY_FOLDER = MAIN_FOLDER + 'sam_boundary_merge/ISPRS_merge_{}.tif'
    OBJECT_FOLDER = MAIN_FOLDER + 'V_merge/V_merge_{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
elif DATASET == 'Urban':
    train_ids = ['1366', '1367', '1368', '1369', '1370', '1371', '1372', '1373', '1374', '1375', '1376', '1377', '1378', '1379', '1380', '1381', '1382', '1383', '1384', '1385', '1386', '1387', '1388', '1389', '1390', '1391', '1392', '1393', '1394', '1395', '1396', '1397', '1398', '1399', '1400', '1401', '1402', '1403', '1404', '1405', '1406', '1407', '1408', '1409', '1410', '1411', '1412', '1413', '1414', '1415', '1416', '1417', '1418', '1419', '1420', '1421', '1422', '1423', '1424', '1425', '1426', '1427', '1428', '1429', '1430', '1431', '1432', '1433', '1434', '1435', '1436', '1437', '1438', '1439', '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448', '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1457', '1458', '1459', '1460', '1461', '1462', '1463', '1464', '1465', '1466', '1467', '1468', '1469', '1470', '1471', '1472', '1473', '1474', '1475', '1476', '1477', '1478', '1479', '1480', '1481', '1482', '1483', '1484', '1485', '1486', '1487', '1488', '1489', '1490', '1491', '1492', '1493', '1494', '1495', '1496', '1497', '1498', '1499', '1500', '1501', '1502', '1503', '1504', '1505', '1506', '1507', '1508', '1509', '1510', '1511', '1512', '1513', '1514', '1515', '1516', '1517', '1518', '1519', '1520', '1521', '1522', '1523', '1524', '1525', '1526', '1527', '1528', '1529', '1530', '1531', '1532', '1533', '1534', '1535', '1536', '1537', '1538', '1539', '1540', '1541', '1542', '1543', '1544', '1545', '1546', '1547', '1548', '1549', '1550', '1551', '1552', '1553', '1554', '1555', '1556', '1557', '1558', '1559', '1560', '1561', '1562', '1563', '1564', '1565', '1566', '1567', '1568', '1569', '1570', '1571', '1572', '1573', '1574', '1575', '1576', '1577', '1578', '1579', '1580', '1581', '1582', '1583', '1584', '1585', '1586', '1587', '1588', '1589', '1590', '1591', '1592', '1593', '1594', '1595', '1596', '1597', '1598', '1599', '1600', '1601', '1602', '1603', '1604', '1605', '1606', '1607', '1608', '1609', '1610', '1611', '1612', '1613', '1614', '1615', '1616', '1617', '1618', '1619', '1620', '1621', '1622', '1623', '1624', '1625', '1626', '1627', '1628', '1629', '1630', '1631', '1632', '1633', '1634', '1635', '1636', '1637', '1638', '1639', '1640', '1641', '1642', '1643', '1644', '1645', '1646', '1647', '1648', '1649', '1650', '1651', '1652', '1653', '1654', '1655', '1656', '1657', '1658', '1659', '1660', '1661', '1662', '1663', '1664', '1665', '1666', '1667', '1668', '1669', '1670', '1671', '1672', '1673', '1674', '1675', '1676', '1677', '1678', '1679', '1680', '1681', '1682', '1683', '1684', '1685', '1686', '1687', '1688', '1689', '1690', '1691', '1692', '1693', '1694', '1695', '1696', '1697', '1698', '1699', '1700', '1701', '1702', '1703', '1704', '1705', '1706', '1707', '1708', '1709', '1710', '1711', '1712', '1713', '1714', '1715', '1716', '1717', '1718', '1719', '1720', '1721', '1722', '1723', '1724', '1725', '1726', '1727', '1728', '1729', '1730', '1731', '1732', '1733', '1734', '1735', '1736', '1737', '1738', '1739', '1740', '1741', '1742', '1743', '1744', '1745', '1746', '1747', '1748', '1749', '1750', '1751', '1752', '1753', '1754', '1755', '1756', '1757', '1758', '1759', '1760', '1761', '1762', '1763', '1764', '1765', '1766', '1767', '1768', '1769', '1770', '1771', '1772', '1773', '1774', '1775', '1776', '1777', '1778', '1779', '1780', '1781', '1782', '1783', '1784', '1785', '1786', '1787', '1788', '1789', '1790', '1791', '1792', '1793', '1794', '1795', '1796', '1797', '1798', '1799', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '1808', '1809', '1810', '1811', '1812', '1813', '1814', '1815', '1816', '1817', '1818', '1819', '1820', '1821', '1822', '1823', '1824', '1825', '1826', '1827', '1828', '1829', '1830', '1831', '1832', '1833', '1834', '1835', '1836', '1837', '1838', '1839', '1840', '1841', '1842', '1843', '1844', '1845', '1846', '1847', '1848', '1849', '1850', '1851', '1852', '1853', '1854', '1855', '1856', '1857', '1858', '1859', '1860', '1861', '1862', '1863', '1864', '1865', '1866', '1867', '1868', '1869', '1870', '1871', '1872', '1873', '1874', '1875', '1876', '1877', '1878', '1879', '1880', '1881', '1882', '1883', '1884', '1885', '1886', '1887', '1888', '1889', '1890', '1891', '1892', '1893', '1894', '1895', '1896', '1897', '1898', '1899', '1900', '1901', '1902', '1903', '1904', '1905', '1906', '1907', '1908', '1909', '1910', '1911', '1912', '1913', '1914', '1915', '1916', '1917', '1918', '1919', '1920', '1921', '1922', '1923', '1924', '1925', '1926', '1927', '1928', '1929', '1930', '1931', '1932', '1933', '1934', '1935', '1936', '1937', '1938', '1939', '1940', '1941', '1942', '1943', '1944', '1945', '1946', '1947', '1948', '1949', '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957', '1958', '1959', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050', '2051', '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059', '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067', '2068', '2069', '2070', '2071', '2072', '2073', '2074', '2075', '2076', '2077', '2078', '2079', '2080', '2081', '2082', '2083', '2084', '2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095', '2096', '2097', '2098', '2099', '2100', '2101', '2102', '2103', '2104', '2105', '2106', '2107', '2108', '2109', '2110', '2111', '2112', '2113', '2114', '2115', '2116', '2117', '2118', '2119', '2120', '2121', '2122', '2123', '2124', '2125', '2126', '2127', '2128', '2129', '2130', '2131', '2132', '2133', '2134', '2135', '2136', '2137', '2138', '2139', '2140', '2141', '2142', '2143', '2144', '2145', '2146', '2147', '2148', '2149', '2150', '2151', '2152', '2153', '2154', '2155', '2156', '2157', '2158', '2159', '2160', '2161', '2162', '2163', '2164', '2165', '2166', '2167', '2168', '2169', '2170', '2171', '2172', '2173', '2174', '2175', '2176', '2177', '2178', '2179', '2180', '2181', '2182', '2183', '2184', '2185', '2186', '2187', '2188', '2189', '2190', '2191', '2192', '2193', '2194', '2195', '2196', '2197', '2198', '2199', '2200', '2201', '2202', '2203', '2204', '2205', '2206', '2207', '2208', '2209', '2210', '2211', '2212', '2213', '2214', '2215', '2216', '2217', '2218', '2219', '2220', '2221', '2222', '2223', '2224', '2225', '2226', '2227', '2228', '2229', '2230', '2231', '2232', '2233', '2234', '2235', '2236', '2237', '2238', '2239', '2240', '2241', '2242', '2243', '2244', '2245', '2246', '2247', '2248', '2249', '2250', '2251', '2252', '2253', '2254', '2255', '2256', '2257', '2258', '2259', '2260', '2261', '2262', '2263', '2264', '2265', '2266', '2267', '2268', '2269', '2270', '2271', '2272', '2273', '2274', '2275', '2276', '2277', '2278', '2279', '2280', '2281', '2282', '2283', '2284', '2285', '2286', '2287', '2288', '2289', '2290', '2291', '2292', '2293', '2294', '2295', '2296', '2297', '2298', '2299', '2300', '2301', '2302', '2303', '2304', '2305', '2306', '2307', '2308', '2309', '2310', '2311', '2312', '2313', '2314', '2315', '2316', '2317', '2318', '2319', '2320', '2321', '2322', '2323', '2324', '2325', '2326', '2327', '2328', '2329', '2330', '2331', '2332', '2333', '2334', '2335', '2336', '2337', '2338', '2339', '2340', '2341', '2342', '2343', '2344', '2345', '2346', '2347', '2348', '2349', '2350', '2351', '2352', '2353', '2354', '2355', '2356', '2357', '2358', '2359', '2360', '2361', '2362', '2363', '2364', '2365', '2366', '2367', '2368', '2369', '2370', '2371', '2372', '2373', '2374', '2375', '2376', '2377', '2378', '2379', '2380', '2381', '2382', '2383', '2384', '2385', '2386', '2387', '2388', '2389', '2390', '2391', '2392', '2393', '2394', '2395', '2396', '2397', '2398', '2399', '2400', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410', '2411', '2412', '2413', '2414', '2415', '2416', '2417', '2418', '2419', '2420', '2421', '2422', '2423', '2424', '2425', '2426', '2427', '2428', '2429', '2430', '2431', '2432', '2433', '2434', '2435', '2436', '2437', '2438', '2439', '2440', '2441', '2442', '2443', '2444', '2445', '2446', '2447', '2448', '2449', '2450', '2451', '2452', '2453', '2454', '2455', '2456', '2457', '2458', '2459', '2460', '2461', '2462', '2463', '2464', '2465', '2466', '2467', '2468', '2469', '2470', '2471', '2472', '2473', '2474', '2475', '2476', '2477', '2478', '2479', '2480', '2481', '2482', '2483', '2484', '2485', '2486', '2487', '2488', '2489', '2490', '2491', '2492', '2493', '2494', '2495', '2496', '2497', '2498', '2499', '2500', '2501', '2502', '2503', '2504', '2505', '2506', '2507', '2508', '2509', '2510', '2511', '2512', '2513', '2514', '2515', '2516', '2517', '2518', '2519', '2520', '2521']
    test_ids = ['3514', '3515', '3516', '3517', '3518', '3519', '3520', '3521', '3522', '3523', '3524', '3525', '3526', '3527', '3528', '3529', '3530', '3531', '3532', '3533', '3534', '3535', '3536', '3537', '3538', '3539', '3540', '3541', '3542', '3543', '3544', '3545', '3546', '3547', '3548', '3549', '3550', '3551', '3552', '3553', '3554', '3555', '3556', '3557', '3558', '3559', '3560', '3561', '3562', '3563', '3564', '3565', '3566', '3567', '3568', '3569', '3570', '3571', '3572', '3573', '3574', '3575', '3576', '3577', '3578', '3579', '3580', '3581', '3582', '3583', '3584', '3585', '3586', '3587', '3588', '3589', '3590', '3591', '3592', '3593', '3594', '3595', '3596', '3597', '3598', '3599', '3600', '3601', '3602', '3603', '3604', '3605', '3606', '3607', '3608', '3609', '3610', '3611', '3612', '3613', '3614', '3615', '3616', '3617', '3618', '3619', '3620', '3621', '3622', '3623', '3624', '3625', '3626', '3627', '3628', '3629', '3630', '3631', '3632', '3633', '3634', '3635', '3636', '3637', '3638', '3639', '3640', '3641', '3642', '3643', '3644', '3645', '3646', '3647', '3648', '3649', '3650', '3651', '3652', '3653', '3654', '3655', '3656', '3657', '3658', '3659', '3660', '3661', '3662', '3663', '3664', '3665', '3666', '3667', '3668', '3669', '3670', '3671', '3672', '3673', '3674', '3675', '3676', '3677', '3678', '3679', '3680', '3681', '3682', '3683', '3684', '3685', '3686', '3687', '3688', '3689', '3690', '3691', '3692', '3693', '3694', '3695', '3696', '3697', '3698', '3699', '3700', '3701', '3702', '3703', '3704', '3705', '3706', '3707', '3708', '3709', '3710', '3711', '3712', '3713', '3714', '3715', '3716', '3717', '3718', '3719', '3720', '3721', '3722', '3723', '3724', '3725', '3726', '3727', '3728', '3729', '3730', '3731', '3732', '3733', '3734', '3735', '3736', '3737', '3738', '3739', '3740', '3741', '3742', '3743', '3744', '3745', '3746', '3747', '3748', '3749', '3750', '3751', '3752', '3753', '3754', '3755', '3756', '3757', '3758', '3759', '3760', '3761', '3762', '3763', '3764', '3765', '3766', '3767', '3768', '3769', '3770', '3771', '3772', '3773', '3774', '3775', '3776', '3777', '3778', '3779', '3780', '3781', '3782', '3783', '3784', '3785', '3786', '3787', '3788', '3789', '3790', '3791', '3792', '3793', '3794', '3795', '3796', '3797', '3798', '3799', '3800', '3801', '3802', '3803', '3804', '3805', '3806', '3807', '3808', '3809', '3810', '3811', '3812', '3813', '3814', '3815', '3816', '3817', '3818', '3819', '3820', '3821', '3822', '3823', '3824', '3825', '3826', '3827', '3828', '3829', '3830', '3831', '3832', '3833', '3834', '3835', '3836', '3837', '3838', '3839', '3840', '3841', '3842', '3843', '3844', '3845', '3846', '3847', '3848', '3849', '3850', '3851', '3852', '3853', '3854', '3855', '3856', '3857', '3858', '3859', '3860', '3861', '3862', '3863', '3864', '3865', '3866', '3867', '3868', '3869', '3870', '3871', '3872', '3873', '3874', '3875', '3876', '3877', '3878', '3879', '3880', '3881', '3882', '3883', '3884', '3885', '3886', '3887', '3888', '3889', '3890', '3891', '3892', '3893', '3894', '3895', '3896', '3897', '3898', '3899', '3900', '3901', '3902', '3903', '3904', '3905', '3906', '3907', '3908', '3909', '3910', '3911', '3912', '3913', '3914', '3915', '3916', '3917', '3918', '3919', '3920', '3921', '3922', '3923', '3924', '3925', '3926', '3927', '3928', '3929', '3930', '3931', '3932', '3933', '3934', '3935', '3936', '3937', '3938', '3939', '3940', '3941', '3942', '3943', '3944', '3945', '3946', '3947', '3948', '3949', '3950', '3951', '3952', '3953', '3954', '3955', '3956', '3957', '3958', '3959', '3960', '3961', '3962', '3963', '3964', '3965', '3966', '3967', '3968', '3969', '3970', '3971', '3972', '3973', '3974', '3975', '3976', '3977', '3978', '3979', '3980', '3981', '3982', '3983', '3984', '3985', '3986', '3987', '3988', '3989', '3990', '3991', '3992', '3993', '3994', '3995', '3996', '3997', '3998', '3999', '4000', '4001', '4002', '4003', '4004', '4005', '4006', '4007', '4008', '4009', '4010', '4011', '4012', '4013', '4014', '4015', '4016', '4017', '4018', '4019', '4020', '4021', '4022', '4023', '4024', '4025', '4026', '4027', '4028', '4029', '4030', '4031', '4032', '4033', '4034', '4035', '4036', '4037', '4038', '4039', '4040', '4041', '4042', '4043', '4044', '4045', '4046', '4047', '4048', '4049', '4050', '4051', '4052', '4053', '4054', '4055', '4056', '4057', '4058', '4059', '4060', '4061', '4062', '4063', '4064', '4065', '4066', '4067', '4068', '4069', '4070', '4071', '4072', '4073', '4074', '4075', '4076', '4077', '4078', '4079', '4080', '4081', '4082', '4083', '4084', '4085', '4086', '4087', '4088', '4089', '4090', '4091', '4092', '4093', '4094', '4095', '4096', '4097', '4098', '4099', '4100', '4101', '4102', '4103', '4104', '4105', '4106', '4107', '4108', '4109', '4110', '4111', '4112', '4113', '4114', '4115', '4116', '4117', '4118', '4119', '4120', '4121', '4122', '4123', '4124', '4125', '4126', '4127', '4128', '4129', '4130', '4131', '4132', '4133', '4134', '4135', '4136', '4137', '4138', '4139', '4140', '4141', '4142', '4143', '4144', '4145', '4146', '4147', '4148', '4149', '4150', '4151', '4152', '4153', '4154', '4155', '4156', '4157', '4158', '4159', '4160', '4161', '4162', '4163', '4164', '4165', '4166', '4167', '4168', '4169', '4170', '4171', '4172', '4173', '4174', '4175', '4176', '4177', '4178', '4179', '4180', '4181', '4182', '4183', '4184', '4185', '4186', '4187', '4188', '4189', '4190']
    Stride_Size = 128
    FOLDER = "/media/lscsc/nas/xianping/ISPRS_dataset/loveDA/"
    LABELS = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    MAIN_FOLDER = FOLDER + 'Urban/'
    DATA_FOLDER = MAIN_FOLDER + 'images_png/{}.png'
    LABEL_FOLDER = MAIN_FOLDER + 'masks_png/{}.png'
    BOUNDARY_FOLDER = MAIN_FOLDER + 'boundary_pngs/{}_Boundary.png'
    OBJECT_FOLDER = MAIN_FOLDER + 'object_pngs/{}_objects.png'
    ERODED_FOLDER = MAIN_FOLDER + 'masks_png/{}.png'
    palette = {0 : (255, 255, 255), # background
           1 : (255, 0, 0),     # building
           2 : (255, 255, 0),   # road
           3 : (0, 0, 255),     # water
           4 : (159, 129, 183), # barren
           5 : (0, 255, 0),     # forest
           6 : (255, 195, 128)} # agriculture
    invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

def object_process(object):
    ids = np.unique(object)
    new_id = 1
    for id in ids[1:]:
        object = np.where(object == id, new_id, object)
        new_id += 1
    return object

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.boundary_files = [BOUNDARY_FOLDER.format(id) for id in ids]
        self.object_files = [OBJECT_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.boundary_cache_ = {}
        self.object_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return BATCH_SIZE * 1000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            ## Potsdam IRRG
            if DATASET == 'Potsdam':
                ## RGB
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                ## IRRG
                # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
            ## Vaihingen IRRG
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
        
        if random_idx in self.boundary_cache_.keys():
            boundary = self.boundary_cache_[random_idx]
        else:
            boundary = np.asarray(io.imread(self.boundary_files[random_idx])) / 255
            boundary = boundary.astype(np.int64)
            if self.cache:
                self.boundary_cache_[random_idx] = boundary

        if random_idx in self.object_cache_.keys():
            object = self.object_cache_[random_idx]
        else:
            object = np.asarray(io.imread(self.object_files[random_idx]))
            object = object.astype(np.int64)
            if self.cache:
                self.object_cache_[random_idx] = object

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            if DATASET == 'Urban':
                label = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64') - 1
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        boundary_p = boundary[x1:x2, y1:y2]
        object_p = object[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        data_p, boundary_p, object_p, label_p = self.data_augmentation(data_p, boundary_p, object_p, label_p)
        object_p = object_process(object_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(boundary_p),
                torch.from_numpy(object_p),
                torch.from_numpy(label_p))
        
## We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss
    
def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()

    return criterion(pred, label, weights)

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class ObjectLoss(nn.Module):
  def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

  def forward(self, pred, gt):
    num_object = int(torch.max(gt)) + 1
    num_object = min(num_object, self.max_object)
    total_object_loss = 0

    for object_index in range(1,num_object):
        mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
        num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
        avg_pool = mask / (num_point + 1)

        object_feature = pred.mul(avg_pool)

        avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1,1,gt.shape[1],gt.shape[2])
        avg_feature = avg_feature.mul(mask)

        object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
        total_object_loss = total_object_loss + object_loss
      
    return total_object_loss
  
class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, _, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)
        class_map = pred.argmax(dim=1).cpu()  # Get Class Map with the Shape: [B, H, W]

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - class_map

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
    
def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return MIoU
