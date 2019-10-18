import pandas as pd
from env import host, user, password

def get_db_url(username, hostname, password, db_name):
    return f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}'

query = '''select parcelid, bedroomcnt, bathroomcnt, fips, calculatedfinishedsquarefeet as squarefeet, taxvaluedollarcnt, taxamount, unitcnt, propertylandusetypeid, propertylandusedesc
	from predictions_2017
	join properties_2017 using(parcelid)
	left join propertylandusetype using(propertylandusetypeid)
	where ((transactiondate like "2017-05%%" or transactiondate like "2017-06%%") and calculatedfinishedsquarefeet is not null);'''

url = get_db_url(user, host, password, 'zillow')

zillow = pd.read_sql(query, url)