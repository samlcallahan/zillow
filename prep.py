import pandas as pd
import numpy as np
from acquire import zillow
import seaborn as sns

# Rename variables
zillow.rename(columns={'bedroomcnt':'beds', 'bathroomcnt':'baths', 'squarefeet':'sqft', 'taxvaluedollarcnt':'value', 'taxamount':'tax', 'unitcnt':'units', 'propertylandusetypeid':'use_id', 'propertylandusedesc':'use'}, inplace=True)

# What is a single-unit property?
all_props = zillow.use.value_counts()
single_unit = zillow[zillow.units == 1].use.value_counts()
percents = all_props / single_unit
null_units = zillow[zillow.units.apply(lambda x: not(x > 0))].use.value_counts()

# Drop non-single unit properties by unit count or by property land use
# If no unit count, drop these: residential general, duplex, quadruplex, triplex, cluster home, commerical/office/residential mixed use
zillow.drop(zillow[zillow.units > 1].index, inplace=True)

nan_mask = zillow.units.apply(lambda x: not(x > 0))
drop_list = [260, 246, 247, 248, 265, 31]
drop_mask = zillow.use_id.apply(lambda x: x in drop_list)
zillow.drop(zillow[drop_mask & nan_mask].index, inplace=True)

# 6037 LA, 6059 Orange, 6111 Ventura
county_dict = {6037:'LA', 6059: 'Orange', 6111: 'Ventura'}
zillow['county'] = zillow.fips.map(county_dict)

# no longer need fips
zillow.drop(columns='fips', inplace=True)

# Drop the one NaN in tax
zillow.tax.dropna(inplace=True)

# box plots to find outliers
sns.boxplot(zillow.beds)
sns.boxplot(zillow.baths)
sns.boxplot(zillow.sqft)
sns.boxplot(zillow.value)

# Drop 0 baths or 0 beds
no_beds_or_baths = zillow[(zillow.beds < 1) | (zillow.baths == 0)]
zillow.drop(no_beds_or_baths.index, inplace=True)

# Drop more than 7 beds
over_7_beds = zillow[zillow.beds > 7]
zillow.drop(over_7_beds.index, inplace=True)

#Drop more than 4.5 baths
over_4ish_baths = zillow[zillow.baths > 4.5]
zillow.drop(over_4ish_baths.index, inplace=True)

# Drop the tiny and huge properties
tiny_props = zillow[zillow.sqft < 300]
zillow.drop(tiny_props.index, inplace=True)

huge_props = zillow[zillow.sqft > 3300]
zillow.drop(huge_props.index, inplace=True)

# Drop homes worth over a million
millions = zillow[zillow.value > 1000000]
zillow.drop(millions.index, inplace=True)