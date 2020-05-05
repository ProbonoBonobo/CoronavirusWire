state = "CA"
county = "Alameda"
fips_index = {}
fips = None
ent = "Sample Enitity"
fips_values = {}

try:
    fips = fips_index[state][county]
except:
    print(f"No fips code for state {state}, county {county}")
fips_values[fips].append(ent)
